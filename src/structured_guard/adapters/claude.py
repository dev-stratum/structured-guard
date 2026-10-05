"""Claude adapter: ``tool_use`` blocks of the Messages API.

Accepts a Messages API response (dict or SDK object), a list of content
blocks, a single ``tool_use`` block, or the JSON text of a tool input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..exceptions import JSONRepairError, RefusalError
from ..models import SchemaLike
from ..sanitizer import as_mapping
from ._common import (
    GuardedCall,
    RawCall,
    make_options,
    run_all,
    run_single,
    text_or_none,
    tool_list,
)

__all__ = ["guard_claude", "guard_claude_calls"]

_TOOL_BLOCKS = ("tool_use", "mcp_tool_use")


def _definitions(tools: Any) -> dict[str, SchemaLike | None]:
    table: dict[str, SchemaLike | None] = {}
    for tool in tool_list(tools):
        name = text_or_none(tool.get("name"))
        if name is not None:
            table[name] = tool.get("input_schema", tool.get("inputSchema"))
    return table


def _blocks_of(payload: Any) -> tuple[list[Any], Any, Any]:
    """Return (content blocks, stop_reason, stop_details)."""
    if isinstance(payload, (list, tuple)):
        return list(payload), None, None
    data = as_mapping(payload)
    if isinstance(data.get("content"), list):
        stop = data.get("stop_reason"), data.get("stop_details")
        return data["content"], *stop
    if data.get("type") in _TOOL_BLOCKS:
        return [data], None, None
    raise JSONRepairError(
        "expected a Claude message, content blocks or a tool_use block"
    )


def _refusal_text(blocks: list[Any], details: Any) -> str:
    texts = [
        block["text"]
        for block in blocks
        if isinstance(block, Mapping) and isinstance(block.get("text"), str)
    ]
    text = " ".join(texts).strip()
    if text:
        return text
    return str(details) if details else "stop_reason is refusal"


def _extract(
    payload: Any, tool_name: str | None
) -> tuple[list[RawCall], str | None]:
    """Return (calls, refusal) for any supported Claude payload."""
    if isinstance(payload, str):
        return [RawCall(tool_name, None, payload)], None
    blocks, stop_reason, details = _blocks_of(payload)
    last = len(blocks) - 1
    calls = [
        RawCall(
            text_or_none(block.get("name")),
            text_or_none(block.get("id")),
            block.get("input"),
            truncated=stop_reason == "max_tokens" and position == last,
        )
        for position, block in enumerate(blocks)
        if isinstance(block, Mapping) and block.get("type") in _TOOL_BLOCKS
    ]
    refusal = None
    if not calls and stop_reason == "refusal":
        refusal = _refusal_text(blocks, details)
    return calls, refusal


def guard_claude(
    payload: Any,
    schema: SchemaLike | None = None,
    *,
    tools: Any = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    index: int = 0,
    **options: Any,
) -> dict[str, Any]:
    """Return the repaired, validated input of one Claude ``tool_use`` call.

    The schema comes from *schema*, or from the matching ``input_schema`` in
    *tools*. When ``stop_reason`` is ``max_tokens`` and the last content
    block is a ``tool_use`` block, that call is treated as cut off. Text is
    taken to be the JSON of one tool input, for example one accumulated
    from a stream. Keyword options are those of
    :func:`structured_guard.inspect_structured_output`.

    Raises:
        RefusalError: If Claude declined and made no tool call.
        ToolCallNotFoundError: If no (defined) tool call matches.
        JSONRepairError: If the input cannot be recovered, or it was cut off
            and ``allow_truncated`` is false.
        SchemaValidationError: If the input violates the schema.
    """
    opts = make_options(options)
    calls, refusal = _extract(payload, tool_name)
    if not calls and refusal is not None:
        raise RefusalError(refusal)
    return run_single(
        calls, schema, tools, _definitions, (tool_name, call_id, index), opts
    )


def guard_claude_calls(
    payload: Any, tools: Any = None, **options: Any
) -> list[GuardedCall]:
    """Guard every ``tool_use`` block, each against its own tool.

    Use it for parallel tool use. See :func:`guard_claude`.
    """
    opts = make_options(options)
    calls, refusal = _extract(payload, None)
    if not calls and refusal is not None:
        raise RefusalError(refusal)
    return run_all(calls, tools, _definitions, opts)
