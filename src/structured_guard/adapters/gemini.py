"""Gemini adapter: ``function_call`` parts and structured output.

Understands the REST spelling (``functionCall``, ``finishReason``) and the
SDK spelling (``function_call``, ``finish_reason``), text-only structured
output, and rebuilds calls from ``MALFORMED_FUNCTION_CALL`` messages. Gemini
schemas use upper-case type names (``OBJECT``); they are normalised to JSON
Schema before validation.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

from ..exceptions import (
    JSONRepairError,
    RefusalError,
    ToolCallNotFoundError,
)
from ..models import Repair, SchemaLike
from ..sanitizer import as_mapping
from ._common import (
    GuardedCall,
    RawCall,
    make_options,
    mark_truncated,
    run_all,
    run_single,
    text_or_none,
    tool_list,
)

__all__ = [
    "guard_gemini",
    "guard_gemini_calls",
    "normalize_gemini_schema",
    "recover_gemini_calls",
]

_JSON_TYPES = frozenset(
    {"null", "boolean", "integer", "number", "string", "array", "object"}
)
_MALFORMED = "MALFORMED_FUNCTION_CALL"
_BLOCKED = frozenset(
    {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
    }
)
_LITERAL_ERRORS = (
    ValueError,
    SyntaxError,
    TypeError,
    RecursionError,
    MemoryError,
)
_CALL_START = re.compile(r"[A-Za-z_][\w.]*\s*\(")
_PREFIX = re.compile(r"^\s*malformed function call\s*:\s*", re.IGNORECASE)
_QUOTE_MAP = str.maketrans(
    {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
)
_RECOVERED = Repair(
    "recovered_call", "rebuilt the call from the model's malformed call text"
)


# -- schemas ----------------------------------------------------------------
def _lower_types(value: Any) -> Any:
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, list):
        return None
    kept = [
        name.lower()
        for name in names
        if isinstance(name, str) and name.lower() in _JSON_TYPES
    ]
    if not kept:
        return None
    return kept[0] if isinstance(value, str) else kept


def normalize_gemini_schema(schema: Any) -> Any:
    """Convert a Gemini function-parameter schema to plain JSON Schema.

    Type names are lower-cased (``"OBJECT"`` becomes ``"object"``) and
    unknown ones such as ``TYPE_UNSPECIFIED`` are dropped, all the way down
    through properties, items and combinators. Other values are unchanged.
    """
    if not isinstance(schema, Mapping):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            lowered = _lower_types(value)
            if lowered is not None:
                out[key] = lowered
        elif key == "properties" and isinstance(value, Mapping):
            out[key] = {
                name: normalize_gemini_schema(sub)
                for name, sub in value.items()
            }
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            out[key] = [normalize_gemini_schema(v) for v in value]
        elif key in ("items", "additionalProperties"):
            out[key] = normalize_gemini_schema(value)
        else:
            out[key] = value
    return out


def _definitions(tools: Any) -> dict[str, SchemaLike | None]:
    table: dict[str, SchemaLike | None] = {}
    for tool in tool_list(tools):
        declared = tool.get("function_declarations")
        if declared is None:
            declared = tool.get("functionDeclarations")
        for declaration in declared if isinstance(declared, list) else [tool]:
            if not isinstance(declaration, Mapping):
                continue
            name = text_or_none(declaration.get("name"))
            if name is None:
                continue
            params = declaration.get("parameters")
            if params is None:
                params = declaration.get(
                    "parametersJsonSchema",
                    declaration.get("parameters_json_schema"),
                )
            table[name] = normalize_gemini_schema(params)
    return table


# -- rebuilding calls from MALFORMED_FUNCTION_CALL text -----------------------
def _paren_end(text: str, start: int) -> int:
    """Index of the parenthesis that closes the one at *start*, or -1."""
    depth, i, n = 0, start, len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"":
            i += 1
            while i < n and text[i] != ch:
                i += 2 if text[i] == "\\" else 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _callee(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _call_in(chunk: str) -> RawCall | None:
    """Parse one Python-style call; only literal keyword arguments count."""
    try:
        tree = ast.parse(chunk, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node.func)
        if name is None or name == "print":
            continue
        if node.args:  # positional arguments cannot be mapped to names
            return None
        try:
            arguments = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
        except _LITERAL_ERRORS:
            return None
        return RawCall(name, None, arguments, repairs=(_RECOVERED,))
    return None


def recover_gemini_calls(message: str) -> list[RawCall]:
    """Rebuild tool calls from the text of a MALFORMED_FUNCTION_CALL.

    Gemini reports the call it tried to make as Python-style text such as
    ``print(default_api.lookup(city="Oslo"))``, sometimes several calls
    run together. The text is parsed with :mod:`ast` (never executed) and
    only literal keyword arguments are accepted.
    """
    text = _PREFIX.sub("", message, count=1).translate(_QUOTE_MAP)
    calls: list[RawCall] = []
    position = 0
    while True:
        match = _CALL_START.search(text, position)
        if match is None:
            break
        end = _paren_end(text, match.end() - 1)
        if end == -1:
            break
        call = _call_in(text[match.start() : end + 1])
        if call is not None:
            calls.append(call)
        position = end + 1
    return calls


# -- payloads -----------------------------------------------------------------
def _function_call(value: Any) -> RawCall | None:
    """Read a ``functionCall`` part or a bare function-call object."""
    if not isinstance(value, Mapping):
        return None
    wrapped = value.get("functionCall", value.get("function_call"))
    if isinstance(wrapped, Mapping):
        source: Mapping[str, Any] = wrapped
    elif "args" in value or "arguments" in value:
        source = value
    else:
        return None
    name = text_or_none(source.get("name"))
    if name is None:
        return None
    return RawCall(
        name,
        text_or_none(source.get("id")),
        source.get("args", source.get("arguments")),
    )


def _parts(parts: Any) -> tuple[list[RawCall], str]:
    calls: list[RawCall] = []
    texts: list[str] = []
    for part in parts if isinstance(parts, list) else []:
        call = _function_call(part)
        if call is not None:
            calls.append(call)
        elif (
            isinstance(part, Mapping)
            and isinstance(part.get("text"), str)
            and not part.get("thought")
        ):
            texts.append(part["text"])
    return calls, "".join(texts)


def _reason(value: Any) -> str:
    return str(getattr(value, "value", value) or "").upper()


def _from_candidate(
    candidate: Mapping[str, Any]
) -> tuple[list[RawCall], str | None, bool]:
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    calls, text = _parts(parts)
    reason = _reason(
        candidate.get("finishReason", candidate.get("finish_reason"))
    )
    message = text_or_none(
        candidate.get("finishMessage", candidate.get("finish_message"))
    )
    refusal = None
    if not calls and text.strip():  # structured output arrives as text
        calls = [RawCall(None, None, text)]
    elif not calls and reason == _MALFORMED:
        calls = recover_gemini_calls(message or "")
        if not calls:
            raise ToolCallNotFoundError(
                "Gemini reported MALFORMED_FUNCTION_CALL and no call could "
                "be rebuilt from its message; retry the request"
            )
    elif not calls and reason in _BLOCKED:
        refusal = f"Gemini blocked the response ({reason})"
    return calls, refusal, reason == "MAX_TOKENS"


def _extract(
    payload: Any, tool_name: str | None
) -> tuple[list[RawCall], str | None, bool]:
    """Return (calls, refusal, cut_off) for any supported Gemini payload."""
    if isinstance(payload, str):
        if payload.lstrip()[:1] not in ("{", "["):
            recovered = recover_gemini_calls(payload)
            if recovered:
                return recovered, None, False
        return [RawCall(tool_name, None, payload)], None, False
    if isinstance(payload, (list, tuple)):
        found: list[RawCall] = []
        for item in payload:
            found.extend(_extract(item, tool_name)[0])
        return found, None, False
    data = as_mapping(payload)
    candidates = data.get("candidates")
    if (
        isinstance(candidates, list)
        and candidates
        and isinstance(candidates[0], Mapping)
    ):
        return _from_candidate(candidates[0])
    if isinstance(data.get("content"), Mapping):  # a candidate
        return _from_candidate(data)
    if isinstance(data.get("parts"), list):  # a Content object
        return _from_candidate({"content": data})
    call = _function_call(data)
    if call is None:
        raise JSONRepairError(
            "expected a Gemini response, part or function call"
        )
    return [call], None, False


def guard_gemini(
    payload: Any,
    schema: SchemaLike | None = None,
    *,
    tools: Any = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    index: int = 0,
    **options: Any,
) -> dict[str, Any]:
    """Return the repaired, validated ``args`` of one Gemini function call.

    *payload* may be a ``generateContent`` response (REST or SDK spelling), a
    candidate, a content object, a list of parts or function calls, or text.
    Text holds either the JSON of the arguments, or the call text of a
    ``MALFORMED_FUNCTION_CALL``. A response that has text but no function
    call is treated as structured output. Gemini integers arrive as floats
    (``3.0``) and are converted when the schema says ``integer``.

    The schema comes from *schema* or from the matching declaration in
    *tools* (``function_declarations`` or ``functionDeclarations``), with
    upper-case type names accepted. Keyword options are those of
    :func:`structured_guard.inspect_structured_output`.

    Raises:
        RefusalError: If the response was blocked and holds no call.
        ToolCallNotFoundError: If no (defined) call matches, or a malformed
            call cannot be rebuilt.
        JSONRepairError: If the arguments cannot be recovered as JSON.
        SchemaValidationError: If the arguments violate the schema.
    """
    opts = make_options(options)
    calls, refusal, cut_off = _extract(payload, tool_name)
    if not calls and refusal is not None:
        raise RefusalError(refusal)
    return run_single(
        mark_truncated(calls, cut_off),
        normalize_gemini_schema(schema),
        tools,
        _definitions,
        (tool_name, call_id, index),
        opts,
    )


def guard_gemini_calls(
    payload: Any, tools: Any = None, **options: Any
) -> list[GuardedCall]:
    """Guard every function call, each against its own declaration.

    Use it for parallel function calling. See :func:`guard_gemini`.
    """
    opts = make_options(options)
    calls, refusal, cut_off = _extract(payload, None)
    if not calls and refusal is not None:
        raise RefusalError(refusal)
    return run_all(mark_truncated(calls, cut_off), tools, _definitions, opts)
