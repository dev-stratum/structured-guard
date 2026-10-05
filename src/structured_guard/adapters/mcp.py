"""MCP adapter: JSON-RPC ``tools/call`` requests and ``CallToolResult``.

Requests are guarded against the tool's ``inputSchema``. Results are
guarded against its ``outputSchema``, using ``structuredContent`` when it is
present and the JSON in the text content otherwise. Payloads may be
dictionaries, SDK objects, or the raw (possibly truncated) JSON text of a
message read from a transport.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from ..exceptions import (
    JSONRepairError,
    ToolCallNotFoundError,
    ToolResultError,
)
from ..models import Repair, SchemaLike
from ..parser import repair_json_text
from ..sanitizer import as_mapping
from ._common import (
    GuardedCall,
    Options,
    RawCall,
    guard_call,
    make_options,
    mark_truncated,
    run_all,
    run_single,
    schema_for,
    text_or_none,
    tool_list,
)

__all__ = ["guard_mcp", "guard_mcp_calls"]

_INPUT_KEYS = ("inputSchema", "input_schema")
_OUTPUT_KEYS = ("outputSchema", "output_schema")
_RESULT_KEYS = (
    "content",
    "structuredContent",
    "structured_content",
    "isError",
    "is_error",
)


def _table(tools: Any, keys: tuple[str, ...]) -> dict[str, SchemaLike | None]:
    table: dict[str, SchemaLike | None] = {}
    for tool in tool_list(tools):
        name = text_or_none(tool.get("name"))
        if name is not None:
            table[name] = next((tool[k] for k in keys if k in tool), None)
    return table


def _inputs(tools: Any) -> dict[str, SchemaLike | None]:
    return _table(tools, _INPUT_KEYS)


def _outputs(tools: Any) -> dict[str, SchemaLike | None]:
    return _table(tools, _OUTPUT_KEYS)


def _request_call(message: Mapping[str, Any]) -> RawCall:
    method = message.get("method")
    if method != "tools/call":
        raise ToolCallNotFoundError(
            f"expected a tools/call request, got the method {method!r}"
        )
    params = message.get("params")
    params = params if isinstance(params, Mapping) else {}
    request_id = message.get("id")
    return RawCall(
        text_or_none(params.get("name")),
        None if request_id is None else str(request_id),
        params.get("arguments"),
    )


def _classify(
    payload: Any, max_depth: int
) -> tuple[str, Any, tuple[Repair, ...], bool]:
    """Return (kind, content, repairs, truncated).

    *kind* is ``"calls"`` (content: a list of RawCall) or ``"result"``
    (content: the CallToolResult mapping).
    """
    repairs: tuple[Repair, ...] = ()
    truncated = False
    if isinstance(payload, str):
        parsed = repair_json_text(payload, max_depth=max_depth)
        payload = parsed.value
        repairs, truncated = parsed.repairs, parsed.truncated
    if isinstance(payload, (list, tuple)):  # a JSON-RPC batch
        batch = [_request_call(m) for m in payload if isinstance(m, Mapping)]
        return "calls", batch, repairs, truncated
    data = as_mapping(payload)
    if "method" in data:
        return "calls", [_request_call(data)], repairs, truncated
    error = data.get("error")
    if isinstance(error, Mapping):
        raise ToolResultError(
            text_or_none(error.get("message")) or "JSON-RPC error"
        )
    if isinstance(data.get("result"), Mapping):
        return "result", data["result"], repairs, truncated
    if any(key in data for key in _RESULT_KEYS):
        return "result", data, repairs, truncated
    if "name" in data:  # the params object of a tools/call request
        call = RawCall(
            text_or_none(data.get("name")), None, data.get("arguments")
        )
        return "calls", [call], repairs, truncated
    raise JSONRepairError("expected a tools/call request or a CallToolResult")


def _result_text(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _with_repairs(call: RawCall, repairs: tuple[Repair, ...]) -> RawCall:
    return dataclasses.replace(call, repairs=call.repairs + repairs)


def _guard_result(
    result: Mapping[str, Any],
    schema: SchemaLike | None,
    tools: Any,
    tool_name: str | None,
    repairs: tuple[Repair, ...],
    truncated: bool,
    options: Options,
) -> dict[str, Any]:
    if result.get("isError") is True or result.get("is_error") is True:
        raise ToolResultError(_result_text(result) or "no details were given")
    resolved = schema_for(tool_name, schema, tools, _outputs)
    structured = result.get(
        "structuredContent", result.get("structured_content")
    )
    arguments = (
        structured if isinstance(structured, Mapping) else _result_text(result)
    )
    raw = RawCall(tool_name, None, arguments, truncated, repairs)
    return guard_call(raw, resolved, options).arguments


def guard_mcp(
    payload: Any,
    schema: SchemaLike | None = None,
    *,
    tools: Any = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    index: int = 0,
    **options: Any,
) -> dict[str, Any]:
    """Return repaired, validated data from an MCP message.

    For a ``tools/call`` request (or a batch of them) the result is the
    ``arguments`` object, checked against the tool's ``inputSchema``. For a
    ``CallToolResult`` (bare, or wrapped in a JSON-RPC response) it is the
    ``structuredContent`` object, or the JSON found in the text content,
    checked against the tool's ``outputSchema``; a result needs *tool_name*
    when the schema is looked up in *tools*. *tools* may be the list from a
    ``tools/list`` result or the whole result. *call_id* matches the
    JSON-RPC ``id``. Keyword options are those of
    :func:`structured_guard.inspect_structured_output`.

    Raises:
        ToolResultError: If the result has ``isError`` set, or the response
            is a JSON-RPC error.
        ToolCallNotFoundError: If the method is not ``tools/call`` or no
            (defined) tool matches.
        JSONRepairError: If the data cannot be recovered as JSON.
        SchemaValidationError: If the data violates the schema.
    """
    opts = make_options(options)
    kind, content, repairs, truncated = _classify(payload, opts.max_depth)
    if kind == "result":
        return _guard_result(
            content, schema, tools, tool_name, repairs, truncated, opts
        )
    calls = mark_truncated(
        [_with_repairs(call, repairs) for call in content], truncated
    )
    return run_single(
        calls, schema, tools, _inputs, (tool_name, call_id, index), opts
    )


def guard_mcp_calls(
    payload: Any, tools: Any = None, **options: Any
) -> list[GuardedCall]:
    """Guard every ``tools/call`` request in *payload* (for example a batch).

    See :func:`guard_mcp`.
    """
    opts = make_options(options)
    kind, content, repairs, truncated = _classify(payload, opts.max_depth)
    if kind == "result":
        raise ToolCallNotFoundError(
            "a CallToolResult is not a tools/call request; use guard_mcp"
        )
    calls = mark_truncated(
        [_with_repairs(call, repairs) for call in content], truncated
    )
    return run_all(calls, tools, _inputs, opts)
