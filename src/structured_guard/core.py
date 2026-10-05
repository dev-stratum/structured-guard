"""Public entry points: raw model output in, checked data out."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import validator as _validator
from .exceptions import (
    JSONRepairError,
    RefusalError,
    SchemaValidationError,
)
from .models import (
    DEFAULT_MAX_DEPTH,
    Repair,
    SchemaLike,
    ValidationIssue,
    summarize,
)
from .parser import repair_json_text
from .sanitizer import read_response

__all__ = [
    "GuardResult",
    "guard_json",
    "inspect_structured_output",
    "repair_structured_output",
]


@dataclass(frozen=True)
class GuardResult:
    """Outcome of checking model output, returned by
    :func:`inspect_structured_output`.

    ``value`` is the repaired and coerced data (``None`` if nothing could be
    parsed). ``errors`` lists what is still wrong, ``repairs`` lists what was
    changed on the way, and ``truncated`` tells whether the JSON had to be
    reconstructed because the output ended early.
    """

    value: Any
    errors: tuple[ValidationIssue, ...] = ()
    repairs: tuple[Repair, ...] = ()
    truncated: bool = False

    @property
    def ok(self) -> bool:
        """True when nothing is wrong with the data."""
        return not self.errors

    def summary(self) -> str:
        """One-line description of the problems, if any."""
        return summarize(self.errors)

    def unwrap(self) -> Any:
        """Return ``value`` or raise :class:`SchemaValidationError`."""
        if self.errors:
            raise SchemaValidationError(self.errors, self.value)
        return self.value

    def retry_prompt(self, *, max_items: int = 10) -> str:
        """Build a follow-up message that tells the model what to fix.

        Returns an empty string when there is nothing to correct.
        """
        if self.ok:
            return ""
        lines = [
            "Your previous reply did not match the required JSON format. "
            "Problems found:"
        ]
        lines.extend(f"- {error}" for error in self.errors[:max_items])
        if len(self.errors) > max_items:
            lines.append(f"- ... and {len(self.errors) - max_items} more")
        if self.truncated:
            lines.append(
                "The reply was cut off before the JSON was complete, "
                "so keep it shorter."
            )
        lines.append(
            "Reply again with only the corrected JSON: no commentary "
            "and no code fences."
        )
        return "\n".join(lines)


def _preferred_root(schema: Mapping[str, Any]) -> str | None:
    kind = schema.get("type")
    if kind in ("object", "array"):
        return str(kind)
    if kind is None and "properties" in schema:
        return "object"
    if kind is None and "items" in schema:
        return "array"
    return None


def _run(
    source: Any,
    schema: SchemaLike | None,
    tool_name: str | None,
    prefer: str | None,
    coerce: bool,
    apply_defaults: bool,
    drop_extra: bool,
    allow_truncated: bool,
    max_depth: int,
    max_issues: int,
) -> GuardResult:
    model_text = read_response(source, tool_name)
    if model_text.refusal is not None:
        raise RefusalError(model_text.refusal)
    if prefer is None and isinstance(schema, Mapping):
        prefer = _preferred_root(schema)
    parsed = repair_json_text(
        model_text.text, prefer=prefer, max_depth=max_depth
    )
    if parsed.truncated and not allow_truncated:
        raise JSONRepairError(
            "the output was cut off before the JSON was complete "
            "(it may have reached the token limit)"
        )
    value = parsed.value
    repairs = list(parsed.repairs)
    errors: list[ValidationIssue] = []
    if schema is not None and coerce:
        outcome = _validator.coerce(
            value,
            schema,
            apply_defaults=apply_defaults,
            drop_extra=drop_extra,
            max_issues=max_issues,
            max_depth=max_depth,
        )
        value = outcome.value
        repairs.extend(outcome.repairs)
        errors.extend(outcome.issues)
    elif schema is not None:
        errors.extend(
            _validator.validate(
                value, schema, max_issues=max_issues, max_depth=max_depth
            )
        )
    return GuardResult(value, tuple(errors), tuple(repairs), parsed.truncated)


def inspect_structured_output(
    source: Any,
    schema: SchemaLike | None = None,
    *,
    tool_name: str | None = None,
    prefer: str | None = None,
    coerce: bool = True,
    apply_defaults: bool = True,
    drop_extra: bool = True,
    allow_truncated: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_issues: int = 50,
) -> GuardResult:
    """Repair and check model output without raising on bad output.

    Unusable output is reported through ``GuardResult.errors`` so a retry
    loop can call :meth:`GuardResult.retry_prompt`. Only a refusal
    (:class:`RefusalError`) and programmer errors, such as an invalid schema,
    raise.

    Args:
        source: Raw text, or a Chat Completions / Responses API response
            (dict or SDK object).
        schema: JSON Schema to check against; ``None`` only repairs.
        tool_name: Read the arguments of this tool call instead of the text.
        prefer: 'object' or 'array' hint; inferred from *schema* if omitted.
        coerce: Apply lossless type conversions.
        apply_defaults: Fill missing properties from schema defaults.
        drop_extra: Drop properties rejected by ``additionalProperties:
            false`` (otherwise they are reported as errors).
        allow_truncated: If false, cut-off output is reported as an error.
        max_depth: Maximum nesting of objects and arrays.
        max_issues: Cap on the number of reported errors.
    """
    try:
        return _run(
            source,
            schema,
            tool_name,
            prefer,
            coerce,
            apply_defaults,
            drop_extra,
            allow_truncated,
            max_depth,
            max_issues,
        )
    except JSONRepairError as error:
        issue = ValidationIssue("parse", f"no usable JSON: {error.message}")
        return GuardResult(None, (issue,))


def repair_structured_output(
    source: Any,
    schema: SchemaLike | None = None,
    *,
    tool_name: str | None = None,
    prefer: str | None = None,
    coerce: bool = True,
    apply_defaults: bool = True,
    drop_extra: bool = True,
    allow_truncated: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_issues: int = 50,
) -> Any:
    """Return the JSON data in *source*, repaired and validated.

    A drop-in replacement for ``json.loads`` on model output. It accepts raw
    text or an OpenAI-style response object; see
    :func:`inspect_structured_output` for the arguments.

    Raises:
        JSONRepairError: If no JSON can be recovered (or the output is
            truncated and *allow_truncated* is false).
        SchemaValidationError: If the data violates *schema*.
        RefusalError: If the model refused to answer.
        InvalidSchemaError: If *schema* itself is invalid.
    """
    return _run(
        source,
        schema,
        tool_name,
        prefer,
        coerce,
        apply_defaults,
        drop_extra,
        allow_truncated,
        max_depth,
        max_issues,
    ).unwrap()


def guard_json(
    func: Callable[..., Any] | None = None,
    *,
    schema: SchemaLike | None = None,
    **options: Any,
) -> Any:
    """Decorator: repair and validate what a function returns.

    Works on plain and ``async`` functions that return model text or an
    API response. Usable bare (``@guard_json``) or with arguments
    (``@guard_json(schema=SCHEMA, tool_name="lookup")``). Extra keyword
    arguments are passed to :func:`repair_structured_output`.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await target(*args, **kwargs)
                return repair_structured_output(result, schema, **options)

            return async_wrapper

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = target(*args, **kwargs)
            return repair_structured_output(result, schema, **options)

        return wrapper

    return decorate if func is None else decorate(func)
