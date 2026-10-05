"""Shared plumbing for the provider adapters.

Every adapter does the same three things: find the tool calls in a provider
payload (``RawCall``), pick the one the caller wants, and run its arguments
through the repair, coercion and validation pipeline (``GuardedCall``).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .. import validator
from ..exceptions import (
    JSONRepairError,
    SchemaValidationError,
    ToolCallNotFoundError,
)
from ..models import DEFAULT_MAX_DEPTH, Repair, SchemaLike, ValidationIssue
from ..parser import repair_json_text

ARGUMENT_KEYS = ("arguments", "parameters", "args", "input")
_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)
_MAX_REENCODINGS = 3

Definitions = Callable[[Any], "dict[str, SchemaLike | None]"]


@dataclass(frozen=True)
class GuardedCall:
    """A tool call whose arguments were repaired and validated."""

    name: str | None
    call_id: str | None
    arguments: dict[str, Any]
    repairs: tuple[Repair, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class RawCall:
    """A tool call as found in a provider payload, before any repair."""

    name: str | None
    call_id: str | None
    arguments: Any
    truncated: bool = False
    repairs: tuple[Repair, ...] = ()


@dataclass(frozen=True)
class Options:
    """Keyword options shared by every ``guard_*`` function."""

    coerce: bool = True
    apply_defaults: bool = True
    drop_extra: bool = True
    allow_truncated: bool = True
    max_depth: int = DEFAULT_MAX_DEPTH
    max_issues: int = 50


def make_options(options: Mapping[str, Any]) -> Options:
    """Build :class:`Options`; an unknown option name raises ``TypeError``."""
    return Options(**options)


def text_or_none(value: Any) -> str | None:
    """Return *value* if it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


def tool_list(tools: Any) -> list[Mapping[str, Any]]:
    """Normalise tool definitions to a list of mappings.

    Accepts a list of definitions, a single definition, or a mapping with a
    ``tools`` list (for example an MCP ``tools/list`` result).
    """
    if isinstance(tools, Mapping):
        inner = tools.get("tools")
        tools = inner if isinstance(inner, list) else [tools]
    if not isinstance(tools, (list, tuple)):
        raise TypeError("tools must be a list of tool definitions")
    return [tool for tool in tools if isinstance(tool, Mapping)]


def schema_for(
    name: str | None,
    schema: SchemaLike | None,
    tools: Any,
    definitions: Definitions,
) -> SchemaLike | None:
    """Pick the schema for a call: the explicit one, else from *tools*."""
    if schema is not None or tools is None:
        return schema
    table = definitions(tools)
    if name is None:
        raise ToolCallNotFoundError(
            "no tool name is known for this payload; pass tool_name= so "
            "its schema can be looked up in tools"
        )
    if name not in table:
        known = ", ".join(sorted(table)) or "none"
        raise ToolCallNotFoundError(
            f"the model called {name!r}, which is not a defined tool "
            f"(defined: {known})"
        )
    return table[name]


def select_call(
    calls: list[RawCall],
    tool_name: str | None,
    call_id: str | None,
    index: int,
) -> RawCall:
    """Pick one call by tool name and/or id, then by position."""
    pool = [
        call
        for call in calls
        if (tool_name is None or call.name == tool_name)
        and (call_id is None or call.call_id == call_id)
    ]
    try:
        return pool[index]
    except IndexError:
        wanted = tool_name if tool_name is not None else "any tool"
        raise ToolCallNotFoundError(
            f"no tool call matched ({wanted}, id {call_id!r}, index "
            f"{index}); the payload has {len(calls)} call(s)"
        ) from None


def mark_truncated(calls: list[RawCall], truncated: bool) -> list[RawCall]:
    """Flag the last call when the provider says generation stopped early."""
    if truncated and calls:
        last = dataclasses.replace(calls[-1], truncated=True)
        return [*calls[:-1], last]
    return calls


def parse_arguments(
    arguments: Any, max_depth: int
) -> tuple[Any, tuple[Repair, ...], bool]:
    """Turn raw tool arguments into Python data.

    Text is repaired as JSON, and JSON that was encoded more than once is
    decoded again. Missing or empty arguments become an empty object, which
    is what a tool without parameters sends.
    """
    if arguments is None or (
        isinstance(arguments, str) and not arguments.strip()
    ):
        note = Repair("empty_arguments", "no arguments; used an empty object")
        return {}, (note,), False
    repairs: list[Repair] = []
    truncated = False
    value = arguments
    rounds = 0
    while isinstance(value, str) and rounds < _MAX_REENCODINGS:
        parsed = repair_json_text(value, prefer="object", max_depth=max_depth)
        if rounds:
            repairs.append(
                Repair("double_encoded", "decoded JSON that was encoded twice")
            )
        repairs.extend(parsed.repairs)
        truncated = truncated or parsed.truncated
        value = parsed.value
        rounds += 1
    if isinstance(value, Mapping):
        value = dict(value)
    return value, tuple(repairs), truncated


def guard_call(
    raw: RawCall, schema: SchemaLike | None, options: Options
) -> GuardedCall:
    """Repair, coerce and validate one call's arguments."""
    value, repairs, truncated = parse_arguments(
        raw.arguments, options.max_depth
    )
    truncated = truncated or raw.truncated
    if truncated and not options.allow_truncated:
        raise JSONRepairError(
            "the tool-call arguments were cut off before they were "
            "complete (the output may have reached the token limit)"
        )
    repairs = raw.repairs + repairs
    effective: SchemaLike = {"type": "object"} if schema is None else schema
    if options.coerce:
        outcome = validator.coerce(
            value,
            effective,
            apply_defaults=options.apply_defaults,
            drop_extra=options.drop_extra,
            max_issues=options.max_issues,
            max_depth=options.max_depth,
        )
        value = outcome.value
        repairs = repairs + outcome.repairs
        issues = list(outcome.issues)
    else:
        issues = list(
            validator.validate(
                value,
                effective,
                max_issues=options.max_issues,
                max_depth=options.max_depth,
            )
        )
    if not issues and not isinstance(value, dict):
        issues.append(
            ValidationIssue("type", "tool arguments must be a JSON object")
        )
    if issues:
        raise SchemaValidationError(issues, value)
    return GuardedCall(raw.name, raw.call_id, value, repairs, truncated)


def run_single(
    calls: list[RawCall],
    schema: SchemaLike | None,
    tools: Any,
    definitions: Definitions,
    selector: tuple[str | None, str | None, int],
    options: Options,
) -> dict[str, Any]:
    """Guard the one call picked by *selector* and return its arguments."""
    raw = select_call(calls, *selector)
    resolved = schema_for(raw.name, schema, tools, definitions)
    return guard_call(raw, resolved, options).arguments


def run_all(
    calls: list[RawCall],
    tools: Any,
    definitions: Definitions,
    options: Options,
) -> list[GuardedCall]:
    """Guard every call, each against its own tool's schema."""
    if not calls:
        raise ToolCallNotFoundError("the payload contains no tool call")
    guarded = []
    for call in calls:
        schema = schema_for(call.name, None, tools, definitions)
        guarded.append(guard_call(call, schema, options))
    return guarded


def envelope_call(item: Any) -> RawCall | None:
    """Read ``{"name": ..., "arguments": {...}}`` style tool calls."""
    if not isinstance(item, Mapping):
        return None
    function = item.get("function")
    source = function if isinstance(function, Mapping) else item
    name = text_or_none(source.get("name"))
    if name is None:
        return None
    for key in ARGUMENT_KEYS:
        if key in source:
            return RawCall(name, text_or_none(item.get("id")), source[key])
    return None


def calls_from_text(text: str, max_depth: int) -> list[RawCall]:
    """Find tool calls that a model wrote into its message text.

    Local models often print calls instead of using the API's structured
    field: a JSON object with ``name`` and ``arguments`` (or ``parameters``),
    a list of such objects, or objects wrapped in ``tool_call`` tags.
    """
    calls: list[RawCall] = []
    for block in _TAGGED.findall(text) or [text]:
        try:
            parsed = repair_json_text(block, max_depth=max_depth)
        except JSONRepairError:
            continue
        value = parsed.value
        for item in value if isinstance(value, list) else [value]:
            call = envelope_call(item)
            if call is not None:
                calls.append(
                    dataclasses.replace(
                        call,
                        truncated=parsed.truncated,
                        repairs=parsed.repairs,
                    )
                )
    return calls
