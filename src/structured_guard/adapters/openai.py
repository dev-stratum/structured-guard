"""OpenAI adapter: function calling, Structured Outputs, compatible APIs.

Reads Chat Completions and Responses API payloads, OpenAI-compatible
servers (vLLM, llama.cpp and similar) and Ollama's native chat response, as
plain dictionaries or as SDK objects that offer ``model_dump()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..exceptions import JSONRepairError, RefusalError
from ..models import SchemaLike
from ..sanitizer import as_mapping, text_of
from ._common import (
    GuardedCall,
    RawCall,
    calls_from_text,
    make_options,
    mark_truncated,
    run_all,
    run_single,
    text_or_none,
    tool_list,
)

__all__ = ["guard_openai", "guard_openai_calls"]

_MESSAGE_KEYS = ("tool_calls", "function_call", "content", "refusal")


def _definitions(tools: Any) -> dict[str, SchemaLike | None]:
    table: dict[str, SchemaLike | None] = {}
    for tool in tool_list(tools):
        function = tool.get("function")
        spec = function if isinstance(function, Mapping) else tool
        name = text_or_none(spec.get("name"))
        if name is not None:
            table[name] = spec.get("parameters")
    return table


def _tool_call(item: Any) -> RawCall | None:
    """Read one Chat Completions tool call or Responses function_call."""
    if not isinstance(item, Mapping):
        return None
    function = item.get("function")
    if isinstance(function, Mapping):
        return RawCall(
            text_or_none(function.get("name")),
            text_or_none(item.get("id")),
            function.get("arguments"),
        )
    if item.get("type") == "function_call" or (
        "name" in item and "arguments" in item
    ):
        call_id = text_or_none(item.get("call_id")) or text_or_none(
            item.get("id")
        )
        return RawCall(
            text_or_none(item.get("name")), call_id, item.get("arguments")
        )
    return None


def _from_text(
    text: str, text_calls: bool, tool_name: str | None, max_depth: int
) -> list[RawCall]:
    """Calls written into text (local models), else the text itself."""
    if text_calls:
        found = calls_from_text(text, max_depth)
        if tool_name is not None:
            found = [call for call in found if call.name == tool_name]
        if found:
            return found
    return [RawCall(tool_name, None, text)]


def _from_message(
    message: Mapping[str, Any],
    text_calls: bool,
    tool_name: str | None,
    max_depth: int,
) -> tuple[list[RawCall], str | None]:
    listed = message.get("tool_calls")
    items = listed if isinstance(listed, list) else []
    calls = [call for call in map(_tool_call, items) if call is not None]
    legacy = message.get("function_call")
    if isinstance(legacy, Mapping) and text_or_none(legacy.get("name")):
        calls.append(
            RawCall(
                text_or_none(legacy.get("name")), None, legacy.get("arguments")
            )
        )
    refusal = text_or_none(message.get("refusal"))
    if not calls and refusal is None:
        text = text_of(message.get("content"))
        if text.strip():
            calls = _from_text(text, text_calls, tool_name, max_depth)
    return calls, refusal


def _from_choices(
    data: Mapping[str, Any],
    text_calls: bool,
    tool_name: str | None,
    max_depth: int,
) -> tuple[list[RawCall], str | None, bool]:
    choices = data["choices"]
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise JSONRepairError("'choices' must be a non-empty list of objects")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise JSONRepairError("the first choice has no 'message' object")
    calls, refusal = _from_message(message, text_calls, tool_name, max_depth)
    return calls, refusal, choice.get("finish_reason") == "length"


def _from_output(
    data: Mapping[str, Any],
    text_calls: bool,
    tool_name: str | None,
    max_depth: int,
) -> tuple[list[RawCall], str | None, bool]:
    items = data["output"]
    if not isinstance(items, list):
        raise JSONRepairError("'output' must be a list")
    calls: list[RawCall] = []
    texts: list[str] = []
    refusal: str | None = None
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "function_call":
            call = _tool_call(item)
            if call is not None:
                calls.append(call)
        elif isinstance(item.get("content"), list):
            for part in item["content"]:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "refusal":
                    refusal = text_or_none(part.get("refusal")) or refusal
                elif isinstance(part.get("text"), str):
                    texts.append(part["text"])
    if not calls and refusal is None and "".join(texts).strip():
        calls = _from_text("".join(texts), text_calls, tool_name, max_depth)
    details = data.get("incomplete_details")
    cut_off = (
        data.get("status") == "incomplete"
        and isinstance(details, Mapping)
        and details.get("reason") == "max_output_tokens"
    )
    return calls, refusal, cut_off


def _extract(
    payload: Any, text_calls: bool, tool_name: str | None, max_depth: int
) -> tuple[list[RawCall], str | None, bool]:
    """Return (calls, refusal, cut_off) for any supported OpenAI payload."""
    if isinstance(payload, str):
        calls = _from_text(payload, text_calls, tool_name, max_depth)
        return calls, None, False
    if isinstance(payload, (list, tuple)):
        found: list[RawCall] = []
        for item in payload:
            found.extend(_extract(item, text_calls, tool_name, max_depth)[0])
        return found, None, False
    data = as_mapping(payload)
    if "choices" in data:
        return _from_choices(data, text_calls, tool_name, max_depth)
    if "output" in data:
        return _from_output(data, text_calls, tool_name, max_depth)
    message = data.get("message")
    if isinstance(message, Mapping):  # Ollama's native chat response
        calls, refusal = _from_message(
            message, text_calls, tool_name, max_depth
        )
        return calls, refusal, data.get("done_reason") == "length"
    if any(key in data for key in _MESSAGE_KEYS):
        calls, refusal = _from_message(data, text_calls, tool_name, max_depth)
        return calls, refusal, False
    call = _tool_call(data)
    if call is None:
        raise JSONRepairError(
            "expected an OpenAI response, message or tool call"
        )
    return [call], None, False


def guard_openai(
    payload: Any,
    schema: SchemaLike | None = None,
    *,
    tools: Any = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    index: int = 0,
    **options: Any,
) -> dict[str, Any]:
    """Return the repaired, validated arguments of one OpenAI tool call.

    *payload* may be a Chat Completions or Responses API response (dict or
    SDK object), a message, one or more tool calls, an Ollama chat response,
    or text. Text holds either the arguments themselves, Structured Output
    JSON, or (when *tools* or *tool_name* is given) tool calls that a local
    model printed as ``{"name": ..., "arguments": ...}``.

    The schema comes from *schema*, or from the matching entry of *tools*
    (a list of function definitions); a call to a tool that *tools* does not
    define raises :class:`ToolCallNotFoundError`. *tool_name*, *call_id* and
    *index* choose among several calls. Other keyword options are those of
    :func:`structured_guard.inspect_structured_output`.

    Raises:
        RefusalError: If the model refused instead of answering.
        ToolCallNotFoundError: If no (defined) tool call matches.
        JSONRepairError: If the arguments cannot be recovered as JSON.
        SchemaValidationError: If the arguments violate the schema.
    """
    opts = make_options(options)
    text_calls = tools is not None or tool_name is not None
    calls, refusal, cut_off = _extract(
        payload, text_calls, tool_name, opts.max_depth
    )
    if not calls and refusal is not None:
        raise RefusalError(refusal)
    return run_single(
        mark_truncated(calls, cut_off),
        schema,
        tools,
        _definitions,
        (tool_name, call_id, index),
        opts,
    )


def guard_openai_calls(
    payload: Any, tools: Any = None, **options: Any
) -> list[GuardedCall]:
    """Guard every tool call in *payload*, each against its own tool.

    Use it for parallel tool calls. See :func:`guard_openai`.
    """
    opts = make_options(options)
    calls, refusal, cut_off = _extract(payload, True, None, opts.max_depth)
    if not calls and refusal is not None:
        raise RefusalError(refusal)
    return run_all(mark_truncated(calls, cut_off), tools, _definitions, opts)
