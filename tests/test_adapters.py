"""Offline tests for the provider adapters, using synthetic payloads only."""

from __future__ import annotations

import copy
import enum
import json
import re

import pytest

import structured_guard as sg
from structured_guard import adapters


def codes(items):
    return [item.code for item in items]


WEATHER = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "minimum": 1},
        "units": {"enum": ["metric", "imperial"], "default": "metric"},
    },
    "required": ["city", "days"],
    "additionalProperties": False,
}
FORECAST = {"city": "Oslo", "days": 3, "units": "metric"}


# ----------------------------------------------------------------- OpenAI
def call(name, arguments, call_id="call_1"):
    function = {"name": name, "arguments": arguments}
    return {"id": call_id, "type": "function", "function": function}


def weather_call(arguments, call_id="call_1"):
    return call("get_weather", arguments, call_id)


def chat(message, finish="tool_calls"):
    message = {"role": "assistant", **message}
    choice = {"index": 0, "finish_reason": finish, "message": message}
    return {"object": "chat.completion", "choices": [choice]}


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": WEATHER},
    },
    {"type": "function", "function": {"name": "ping"}},
]


def test_openai_function_call_arguments_are_repaired_and_coerced():
    arguments = '{"city": "Oslo", "days": "3",}'
    response = chat({"tool_calls": [weather_call(arguments)]})
    assert adapters.guard_openai(response, WEATHER) == FORECAST


def test_openai_truncated_arguments_are_reconstructed():
    arguments = '{"city": "Oslo", "days": 3'
    cut = chat({"tool_calls": [weather_call(arguments)]}, "length")
    assert adapters.guard_openai(cut, WEATHER) == FORECAST
    (guarded,) = adapters.guard_openai_calls(cut)
    assert guarded.truncated is True
    assert "truncated" in codes(guarded.repairs)
    with pytest.raises(sg.JSONRepairError, match="cut off"):
        adapters.guard_openai(cut, WEATHER, allow_truncated=False)


def test_openai_length_finish_reason_flags_complete_looking_arguments():
    arguments = '{"city": "Oslo", "days": 3}'
    cut = chat({"tool_calls": [weather_call(arguments)]}, "length")
    assert adapters.guard_openai_calls(cut)[0].truncated is True
    whole = chat({"tool_calls": [weather_call(arguments)]})
    assert adapters.guard_openai_calls(whole)[0].truncated is False


def test_openai_parallel_calls_are_selected_by_name_id_and_position():
    parallel = chat(
        {
            "tool_calls": [
                weather_call('{"city": "Oslo", "days": 1}', "call_a"),
                weather_call('{"city": "Rome", "days": 2}', "call_b"),
                call("ping", "{}", "call_c"),
            ]
        }
    )
    pick = adapters.guard_openai
    second = pick(parallel, WEATHER, tool_name="get_weather", index=1)
    assert second["city"] == "Rome"
    assert pick(parallel, WEATHER, call_id="call_a")["city"] == "Oslo"
    assert pick(parallel, tool_name="ping") == {}
    with pytest.raises(sg.ToolCallNotFoundError):
        pick(parallel, tool_name="missing")
    with pytest.raises(sg.ToolCallNotFoundError, match="index 2"):
        pick(parallel, WEATHER, tool_name="get_weather", index=2)


def test_openai_each_call_is_checked_against_its_own_tool():
    parallel = chat(
        {
            "tool_calls": [
                weather_call('{"city": "Oslo", "days": "2"}', "a"),
                call("ping", "", "b"),
            ]
        }
    )
    first, second = adapters.guard_openai_calls(parallel, OPENAI_TOOLS)
    assert (first.name, first.call_id) == ("get_weather", "a")
    assert first.arguments == {"city": "Oslo", "days": 2, "units": "metric"}
    assert (second.name, second.arguments) == ("ping", {})
    flat = [{"type": "function", "name": "ping", "parameters": WEATHER}]
    ping = chat({"tool_calls": [call("ping", "{}")]})
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_openai(ping, tools=flat)


def test_openai_a_call_to_an_undefined_tool_is_rejected():
    response = chat({"tool_calls": [call("delete_everything", "{}")]})
    with pytest.raises(sg.ToolCallNotFoundError, match="delete_everything"):
        adapters.guard_openai(response, tools=OPENAI_TOOLS)
    with pytest.raises(TypeError):
        adapters.guard_openai(response, tools=5)


def test_openai_structured_output_text_is_unwrapped_and_validated():
    text = 'Sure!\n```json\n{"city": "Oslo", "days": "3"}\n```'
    response = chat({"content": text}, "stop")
    assert adapters.guard_openai(response, WEATHER) == FORECAST
    parts = [
        {"type": "text", "text": '{"city": "Oslo", '},
        "junk",
        {"type": "text", "text": '"days": 3}'},
    ]
    response = chat({"content": parts}, "stop")
    assert adapters.guard_openai(response, WEATHER) == FORECAST


def test_openai_structured_output_cut_off_by_the_token_limit():
    cut = chat({"content": '{"city": "Oslo", "days": 2'}, "length")
    assert adapters.guard_openai(cut, WEATHER)["days"] == 2
    text = '{"city": "Oslo", "days": 2, "units": "met'
    bad = chat({"content": text}, "length")
    with pytest.raises(sg.SchemaValidationError) as info:
        adapters.guard_openai(bad, WEATHER)
    assert info.value.errors[0].pointer == "/units"


def test_openai_refusals_raise_refusal_error():
    refused = chat({"content": None, "refusal": "I can't help."}, "stop")
    with pytest.raises(sg.RefusalError, match="can't help"):
        adapters.guard_openai(refused, WEATHER)
    with pytest.raises(sg.RefusalError):
        adapters.guard_openai_calls(refused)


def test_openai_responses_api_function_call_items():
    arguments = '{"city": "Oslo", "days": "3"}'
    response = {
        "status": "completed",
        "output": [
            "junk",
            {"type": "reasoning", "summary": []},
            {
                "type": "function_call",
                "call_id": "call_9",
                "name": "get_weather",
                "arguments": arguments,
            },
        ],
    }
    assert adapters.guard_openai(response, WEATHER) == FORECAST
    (guarded,) = adapters.guard_openai_calls(response, OPENAI_TOOLS)
    assert (guarded.name, guarded.call_id) == ("get_weather", "call_9")


def test_openai_responses_api_structured_output_refusal_and_limit():
    def reply(*parts):
        return {"type": "message", "content": list(parts)}

    whole = {"type": "output_text", "text": '{"city": "Oslo", "days": 3}'}
    done = {"output": [reply("junk", whole)]}
    assert adapters.guard_openai(done, WEATHER) == FORECAST
    refusal = {"type": "refusal", "refusal": "No."}
    with pytest.raises(sg.RefusalError, match="No."):
        adapters.guard_openai({"output": [reply(refusal)]})
    limit = {"reason": "max_output_tokens"}
    cut = {"status": "incomplete", "incomplete_details": limit}
    cut["output"] = [reply(whole)]
    with pytest.raises(sg.JSONRepairError, match="cut off"):
        adapters.guard_openai(cut, WEATHER, allow_truncated=False)
    other = {**cut, "incomplete_details": {"reason": "content_filter"}}
    assert adapters.guard_openai_calls(other)[0].truncated is False
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_openai({"output": "nope"})


def test_openai_legacy_function_call_field():
    legacy = {
        "name": "get_weather",
        "arguments": '{"city": "Oslo", "days": 5}',
    }
    response = chat({"content": None, "function_call": legacy}, "stop")
    assert adapters.guard_openai(response, WEATHER)["days"] == 5


def test_openai_compatible_servers_and_ollama():
    quoted = "{'city': 'Oslo', 'days': '3'}"
    vllm = chat({"tool_calls": [weather_call(quoted, "chatcmpl-tool-1")]})
    assert adapters.guard_openai(vllm, WEATHER) == FORECAST
    function = {
        "name": "get_weather",
        "arguments": {"city": "Oslo", "days": "3"},
    }
    ollama = {
        "model": "local-model",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": function}],
        },
        "done": True,
        "done_reason": "stop",
    }
    assert adapters.guard_openai(ollama, WEATHER) == FORECAST
    ollama["done_reason"] = "length"
    assert adapters.guard_openai_calls(ollama)[0].truncated is True


def test_openai_local_models_print_tool_calls_as_text():
    llama = (
        '{"name": "get_weather", '
        '"parameters": {"city": "Oslo", "days": 2}}'
    )
    got = adapters.guard_openai(
        chat({"content": llama}, "stop"), WEATHER, tool_name="get_weather"
    )
    assert got == {"city": "Oslo", "days": 2, "units": "metric"}
    tagged = (
        "I will check both.\n"
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Oslo", "days": 1}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}\n'
        "</tool_call>"
    )
    found = adapters.guard_openai_calls(tagged, OPENAI_TOOLS)
    assert [c.arguments["city"] for c in found] == ["Oslo", "Rome"]
    second = adapters.guard_openai(tagged, tools=OPENAI_TOOLS, index=1)
    assert second["city"] == "Rome"
    mistral = (
        '[TOOL_CALLS] [{"name": "get_weather", '
        '"arguments": {"city": "Oslo", "days": 3}}]'
    )
    assert adapters.guard_openai(mistral, tools=OPENAI_TOOLS) == FORECAST
    cut = (
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"city": "Oslo", "days": 2'
    )
    (guarded,) = adapters.guard_openai_calls(cut, OPENAI_TOOLS)
    assert guarded.truncated and guarded.arguments["days"] == 2


def test_openai_text_is_arguments_unless_a_tool_is_named():
    envelope = (
        '{"name": "get_weather", "parameters": {"city": "Oslo", "days": 3}}'
    )
    plain = adapters.guard_openai(chat({"content": envelope}, "stop"))
    assert plain["name"] == "get_weather"
    arguments = '{"city": "Oslo", "days": 2}'
    named = adapters.guard_openai(arguments, WEATHER, tool_name="get_weather")
    assert named["days"] == 2
    odd = '{"name": "x", "input": "y"}'
    search = {
        "properties": {"name": {"type": "string"}, "input": {"type": "string"}}
    }
    got = adapters.guard_openai(odd, search, tool_name="search")
    assert got == {"name": "x", "input": "y"}
    with pytest.raises(sg.ToolCallNotFoundError, match="tool_name"):
        adapters.guard_openai("{}", tools=OPENAI_TOOLS)


def test_openai_text_that_is_not_a_tool_call_is_rejected_cleanly():
    for text in (
        "just thinking out loud",
        "[1, 2]",
        '{"name": "get_weather"}',
        '{"city": "Oslo"}',
        "<tool_call>\nnot json\n</tool_call>",
    ):
        with pytest.raises(sg.ToolCallNotFoundError, match="tool_name"):
            adapters.guard_openai(text, tools=OPENAI_TOOLS)


def test_openai_double_encoded_and_empty_arguments():
    encoded = json.dumps(json.dumps({"city": "Oslo", "days": 3}))
    response = chat({"tool_calls": [weather_call(encoded)]})
    assert adapters.guard_openai(response, WEATHER) == FORECAST
    (guarded,) = adapters.guard_openai_calls(response)
    assert "double_encoded" in codes(guarded.repairs)
    for empty in ("", "  ", None):
        response = chat({"tool_calls": [call("ping", empty)]})
        assert adapters.guard_openai(response) == {}
    levels = json.dumps({"a": 1})
    for _ in range(3):
        levels = json.dumps(levels)
    four = chat({"tool_calls": [call("ping", levels)]})
    assert adapters.guard_openai(four) == {"a": 1}
    five = chat({"tool_calls": [call("ping", json.dumps(levels))]})
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_openai(five)


def test_openai_schema_violations_carry_details():
    arguments = '{"city": "Oslo", "extra": 1}'
    response = chat({"tool_calls": [weather_call(arguments)]})
    with pytest.raises(sg.SchemaValidationError) as info:
        adapters.guard_openai(response, WEATHER)
    assert [e.pointer for e in info.value.errors] == ["/days"]
    assert info.value.value == {"city": "Oslo", "units": "metric"}
    listed = chat({"tool_calls": [weather_call("[1, 2]")]})
    with pytest.raises(sg.SchemaValidationError, match="expected object"):
        adapters.guard_openai(listed)
    with pytest.raises(sg.SchemaValidationError, match="JSON object"):
        adapters.guard_openai(listed, True)
    broken = chat({"tool_calls": [call("ping", "no json")]})
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_openai(broken)


def test_openai_options_are_validated_and_forwarded():
    arguments = '{"city": "Oslo", "days": "3", "x": 1}'
    response = chat({"tool_calls": [weather_call(arguments)]})
    with pytest.raises(TypeError):
        adapters.guard_openai(response, WEATHER, colour="blue")
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_openai(response, WEATHER, coerce=False)
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_openai(response, WEATHER, drop_extra=False)
    plain = adapters.guard_openai(response, WEATHER, apply_defaults=False)
    assert plain == {"city": "Oslo", "days": 3}
    with pytest.raises(ValueError, match="max_depth"):
        adapters.guard_openai(response, WEATHER, max_depth=0)


class OpenAIObject:
    def model_dump(self):
        arguments = '{"city": "Oslo", "days": 3}'
        return chat({"tool_calls": [weather_call(arguments)]})


def test_openai_accepts_sdk_objects_lists_and_single_calls():
    assert adapters.guard_openai(OpenAIObject(), WEATHER) == FORECAST
    single = weather_call('{"city": "Oslo", "days": 3}')
    assert adapters.guard_openai(single, WEATHER) == FORECAST
    assert adapters.guard_openai([single, "junk"], WEATHER) == FORECAST
    bare = {"name": "get_weather", "arguments": '{"city": "Oslo", "days": 3}'}
    assert adapters.guard_openai(bare, WEATHER) == FORECAST
    message = {"role": "assistant", "tool_calls": [single]}
    assert adapters.guard_openai(message, WEATHER) == FORECAST
    mixed = chat({"tool_calls": ["junk", single]})
    assert adapters.guard_openai(mixed, WEATHER) == FORECAST
    odd = chat({"tool_calls": 5, "content": '{"city": "Oslo", "days": 3}'})
    assert adapters.guard_openai(odd, WEATHER) == FORECAST


@pytest.mark.parametrize(
    "payload",
    [
        None,
        5,
        {},
        {"hello": 1},
        {"choices": []},
        {"choices": "x"},
        {"choices": ["x"]},
        {"choices": [{}]},
        {"choices": [{"message": "x"}]},
    ],
)
def test_openai_unrecognised_payloads_raise(payload):
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_openai(payload)


# ----------------------------------------------------------------- Claude
TRIP = {
    "type": "object",
    "properties": {
        "stops": {"type": "array", "items": {"type": "string"}},
        "budget": {
            "type": "object",
            "properties": {"max": {"type": "number"}},
        },
        "days": {"type": "integer"},
    },
    "required": ["stops"],
}
CLAUDE_TOOLS = [
    {"name": "plan_trip", "description": "Plan a trip.", "input_schema": TRIP},
    {"name": "ping", "inputSchema": {"type": "object"}},
]


def tool_use(name, tool_input, tool_id="toolu_01"):
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": tool_input,
    }


def reply(blocks, stop="tool_use"):
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def test_claude_tool_use_with_json_encoded_values_is_coerced():
    values = {
        "stops": '["Oslo", "Bergen"]',
        "budget": '{"max": "250.5"}',
        "days": "3",
    }
    blocks = [
        {"type": "text", "text": "Planning."},
        tool_use("plan_trip", values),
    ]
    assert adapters.guard_claude(reply(blocks), TRIP) == {
        "stops": ["Oslo", "Bergen"],
        "budget": {"max": 250.5},
        "days": 3,
    }


def test_claude_missing_input_and_streamed_input_text():
    block = {"type": "tool_use", "id": "toolu_02", "name": "ping"}
    assert adapters.guard_claude(reply([block])) == {}
    streamed = '{"stops": ["Oslo", "Ber'
    assert adapters.guard_claude(streamed, TRIP)["stops"] == ["Oslo", "Ber"]
    (guarded,) = adapters.guard_claude_calls(streamed)
    assert guarded.truncated and guarded.name is None


def test_claude_max_tokens_cuts_off_only_a_trailing_tool_use():
    blocks = [
        tool_use("plan_trip", {"stops": ["a"]}, "t1"),
        tool_use("plan_trip", {"stops": []}, "t2"),
    ]
    cut = reply(blocks, "max_tokens")
    first, second = adapters.guard_claude_calls(cut, CLAUDE_TOOLS)
    assert (first.truncated, second.truncated) == (False, True)
    with pytest.raises(sg.JSONRepairError, match="cut off"):
        adapters.guard_claude(cut, TRIP, index=1, allow_truncated=False)
    kept = adapters.guard_claude(cut, TRIP, index=0, allow_truncated=False)
    assert kept == {"stops": ["a"]}
    text_last = reply(blocks + [{"type": "text", "text": "cut"}], "max_tokens")
    assert not any(c.truncated for c in adapters.guard_claude_calls(text_last))


def test_claude_parallel_tool_use_and_definitions():
    blocks = [
        tool_use("plan_trip", {"stops": ["Oslo"]}, "t1"),
        tool_use("ping", {}, "t2"),
    ]
    parallel = reply(blocks)
    assert adapters.guard_claude(parallel, call_id="t2") == {}
    got = adapters.guard_claude(
        parallel, tools=CLAUDE_TOOLS, tool_name="plan_trip"
    )
    assert got == {"stops": ["Oslo"]}
    found = adapters.guard_claude_calls(parallel, CLAUDE_TOOLS)
    assert [c.name for c in found] == ["plan_trip", "ping"]
    rogue = reply([tool_use("rm_rf", {})])
    with pytest.raises(sg.ToolCallNotFoundError, match="rm_rf"):
        adapters.guard_claude(rogue, tools=CLAUDE_TOOLS)
    missing = reply([tool_use("plan_trip", {"budget": {}})])
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_claude(missing, tools=CLAUDE_TOOLS)


def test_claude_no_tool_use_and_refusals():
    talk = reply([{"type": "text", "text": "Hello."}], "end_turn")
    with pytest.raises(sg.ToolCallNotFoundError):
        adapters.guard_claude(talk)
    with pytest.raises(sg.ToolCallNotFoundError, match="no tool call"):
        adapters.guard_claude_calls(talk)
    said = reply([{"type": "text", "text": "I can't do that."}], "refusal")
    with pytest.raises(sg.RefusalError, match="can't do that"):
        adapters.guard_claude(said)
    detailed = {**reply([], "refusal"), "stop_details": {"type": "refusal"}}
    with pytest.raises(sg.RefusalError, match="type"):
        adapters.guard_claude_calls(detailed)
    with pytest.raises(sg.RefusalError, match="stop_reason"):
        adapters.guard_claude(reply([], "refusal"))


class ClaudeObject:
    def model_dump(self):
        return reply([tool_use("plan_trip", {"stops": ["Oslo"]})])


def test_claude_other_payload_shapes():
    block = tool_use("plan_trip", {"stops": ["Oslo"]})
    assert adapters.guard_claude(block, TRIP) == {"stops": ["Oslo"]}
    assert adapters.guard_claude([block, "junk"], TRIP) == {"stops": ["Oslo"]}
    connector = {
        "type": "mcp_tool_use",
        "id": "m1",
        "name": "x",
        "server_name": "s",
        "input": '{"stops": ["a",]}',
    }
    assert adapters.guard_claude(reply([connector]), TRIP) == {"stops": ["a"]}
    assert adapters.guard_claude(ClaudeObject(), TRIP) == {"stops": ["Oslo"]}


@pytest.mark.parametrize(
    "payload", [None, 5, {}, {"hello": 1}, {"content": "text"}]
)
def test_claude_unrecognised_payloads_raise(payload):
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_claude(payload)


# -------------------------------------------------------------------- MCP
READ_INPUT = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "max_bytes": {"type": "integer", "minimum": 1},
    },
    "required": ["path"],
    "additionalProperties": False,
}
READ_OUTPUT = {
    "type": "object",
    "properties": {
        "lines": {"type": "integer"},
        "preview": {"type": "string"},
    },
    "required": ["lines"],
}
MCP_TOOLS = {
    "tools": [
        {
            "name": "read_file",
            "description": "Read a file.",
            "inputSchema": READ_INPUT,
            "outputSchema": READ_OUTPUT,
        },
        {"name": "ping", "inputSchema": {"type": "object"}},
    ]
}


def rpc(name, arguments, request_id=1):
    params = {"name": name, "arguments": arguments}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": params,
    }


def test_mcp_tools_call_request_is_repaired_and_validated():
    text = '{"path": "./notes.txt", "max_bytes": "512"}'
    expected = {"path": "./notes.txt", "max_bytes": 512}
    request = rpc("read_file", text)
    assert adapters.guard_mcp(request, tools=MCP_TOOLS) == expected
    parsed = rpc("read_file", json.loads(text))
    assert adapters.guard_mcp(parsed, READ_INPUT) == expected
    numbered = rpc("read_file", text, 41)
    (guarded,) = adapters.guard_mcp_calls(numbered, MCP_TOOLS)
    assert (guarded.name, guarded.call_id) == ("read_file", "41")
    invalid = rpc("read_file", {"max_bytes": 0})
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_mcp(invalid, tools=MCP_TOOLS)


def test_mcp_truncated_transport_line_is_recovered():
    line = (
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":'
        '{"name":"read_file","arguments":{"path":"./notes.txt"'
    )
    assert adapters.guard_mcp(line, tools=MCP_TOOLS) == {"path": "./notes.txt"}
    (guarded,) = adapters.guard_mcp_calls(line, MCP_TOOLS)
    assert guarded.truncated and guarded.call_id == "7"
    with pytest.raises(sg.JSONRepairError, match="cut off"):
        adapters.guard_mcp(line, tools=MCP_TOOLS, allow_truncated=False)
    noisy = "log: " + json.dumps(rpc("ping", {})) + " <- sent"
    assert adapters.guard_mcp(noisy, tools=MCP_TOOLS) == {}


def test_mcp_batches_and_selection():
    batch = [
        rpc("read_file", {"path": "./a.txt"}, 1),
        "junk",
        rpc("ping", {}, 2),
    ]
    both = adapters.guard_mcp_calls(batch, MCP_TOOLS)
    assert [c.name for c in both] == ["read_file", "ping"]
    assert adapters.guard_mcp(batch, call_id="2", tools=MCP_TOOLS) == {}
    named = adapters.guard_mcp(batch, tools=MCP_TOOLS, tool_name="read_file")
    assert named == {"path": "./a.txt"}


def test_mcp_call_tool_result_with_structured_content():
    result = {
        "content": [{"type": "text", "text": '{"lines": 3}'}],
        "structuredContent": {"lines": "3", "preview": "a"},
        "isError": False,
    }
    got = adapters.guard_mcp(result, tools=MCP_TOOLS, tool_name="read_file")
    assert got == {"lines": 3, "preview": "a"}
    wrapped = {"jsonrpc": "2.0", "id": 1, "result": result}
    got = adapters.guard_mcp(wrapped, READ_OUTPUT)
    assert got == {"lines": 3, "preview": "a"}
    broken = {"structuredContent": {"preview": "a"}}
    with pytest.raises(sg.SchemaValidationError) as info:
        adapters.guard_mcp(broken, READ_OUTPUT)
    assert info.value.errors[0].pointer == "/lines"


def test_mcp_call_tool_result_falls_back_to_text_content():
    text = '{"lines": 2, "preview": "x",}'
    result = {"content": [{"type": "image"}, {"type": "text", "text": text}]}
    got = adapters.guard_mcp(result, READ_OUTPUT)
    assert got == {"lines": 2, "preview": "x"}
    partial = '{"lines": 2, "preview": "partial'
    cut = {"content": [{"type": "text", "text": partial}]}
    assert adapters.guard_mcp(cut, READ_OUTPUT)["preview"] == "partial"
    prose = {"content": [{"type": "text", "text": "File saved."}]}
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_mcp(prose)
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_mcp({"content": []}, READ_OUTPUT)
    assert adapters.guard_mcp({"content": "x", "isError": False}) == {}
    snake = {"structured_content": {"lines": 1}}
    assert adapters.guard_mcp(snake, READ_OUTPUT) == {"lines": 1}


def test_mcp_error_results_and_json_rpc_errors():
    block = {"type": "text", "text": "disk full"}
    failed = {"content": [block], "isError": True}
    with pytest.raises(sg.ToolResultError, match="disk full") as info:
        adapters.guard_mcp(failed, READ_OUTPUT)
    assert info.value.message == "disk full"
    with pytest.raises(sg.ToolResultError, match="no details"):
        adapters.guard_mcp({"is_error": True})
    error = {"code": -32602, "message": "Unknown tool: nope"}
    protocol = {"jsonrpc": "2.0", "id": 1, "error": error}
    with pytest.raises(sg.ToolResultError, match="Unknown tool"):
        adapters.guard_mcp(protocol)
    bare = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603}}
    with pytest.raises(sg.ToolResultError, match="JSON-RPC error"):
        adapters.guard_mcp(bare)


def test_mcp_tool_definitions_and_output_schemas():
    as_list = MCP_TOOLS["tools"]
    assert adapters.guard_mcp(rpc("ping", {}), tools=as_list) == {}
    snake = [
        {
            "name": "read_file",
            "input_schema": READ_INPUT,
            "output_schema": READ_OUTPUT,
        }
    ]
    request = rpc("read_file", {"path": "./x"})
    assert adapters.guard_mcp(request, tools=snake) == {"path": "./x"}
    result = {"structuredContent": {"lines": 1}}
    got = adapters.guard_mcp(result, tools=snake, tool_name="read_file")
    assert got == {"lines": 1}
    with pytest.raises(sg.ToolCallNotFoundError, match="tool_name"):
        adapters.guard_mcp(result, tools=snake)
    with pytest.raises(sg.ToolCallNotFoundError, match="nope"):
        adapters.guard_mcp(rpc("nope", {}), tools=snake)
    no_output = [{"name": "ping", "inputSchema": {"type": "object"}}]
    free = {"structuredContent": {"a": 1}}
    got = adapters.guard_mcp(free, tools=no_output, tool_name="ping")
    assert got == {"a": 1}


def test_mcp_wrong_methods_and_unknown_shapes():
    listing = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    with pytest.raises(sg.ToolCallNotFoundError, match="tools/list"):
        adapters.guard_mcp(listing)
    params_only = {"name": "read_file", "arguments": '{"path": "./y"}'}
    assert adapters.guard_mcp(params_only, READ_INPUT) == {"path": "./y"}
    odd = {"jsonrpc": "2.0", "method": "tools/call", "params": []}
    assert adapters.guard_mcp(odd) == {}
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_mcp({"hello": 1})
    with pytest.raises(sg.ToolCallNotFoundError, match="CallToolResult"):
        adapters.guard_mcp_calls({"structuredContent": {"a": 1}})


# ----------------------------------------------------------------- Gemini
GEMINI_PARAMS = {
    "type": "OBJECT",
    "properties": {
        "city": {"type": "STRING"},
        "days": {"type": "INTEGER"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "note": {"type": "STRING", "nullable": True},
    },
    "required": ["city", "days"],
}
GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Weather.",
                "parameters": GEMINI_PARAMS,
            },
            {"name": "ping"},
            "junk",
            {"description": "nameless"},
        ]
    }
]


def candidate(parts, finish="STOP", finish_message=None):
    body = {
        "content": {"role": "model", "parts": parts},
        "finishReason": finish,
        "index": 0,
    }
    if finish_message is not None:
        body["finishMessage"] = finish_message
    return {"candidates": [body], "usageMetadata": {"promptTokenCount": 5}}


def gemini_call(name, args, **extra):
    return {"functionCall": {"name": name, "args": args, **extra}}


def test_gemini_rest_function_call_with_float_integers():
    part = gemini_call("get_weather", {"city": "Oslo", "days": 3.0})
    response = candidate([part])
    got = adapters.guard_gemini(response, GEMINI_PARAMS)
    assert got == {"city": "Oslo", "days": 3}
    assert isinstance(got["days"], int)
    (guarded,) = adapters.guard_gemini_calls(response, GEMINI_TOOLS)
    assert "coerced_type" in codes(guarded.repairs)


class Reason(enum.Enum):
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"


def test_gemini_sdk_spelling_and_enum_finish_reasons():
    function = {
        "name": "get_weather",
        "id": "fc_1",
        "args": {"city": "Oslo", "days": "3"},
    }
    part = {"function_call": function}
    content = {"parts": [part]}
    sdk = {"candidates": [{"content": content, "finish_reason": Reason.STOP}]}
    got = adapters.guard_gemini(sdk, GEMINI_PARAMS)
    assert got == {"city": "Oslo", "days": 3}
    assert adapters.guard_gemini_calls(sdk)[0].call_id == "fc_1"
    last = {"content": content, "finish_reason": Reason.MAX_TOKENS}
    cut = {"candidates": [last]}
    assert adapters.guard_gemini_calls(cut)[0].truncated is True
    bare = {"name": "get_weather", "args": {"city": "Oslo", "days": 2}}
    assert adapters.guard_gemini([bare], GEMINI_PARAMS)["days"] == 2
    assert adapters.guard_gemini(content, GEMINI_PARAMS)["city"] == "Oslo"
    wrapped = {"content": content}
    assert adapters.guard_gemini(wrapped, GEMINI_PARAMS)["city"] == "Oslo"
    assert adapters.guard_gemini(part, GEMINI_PARAMS)["city"] == "Oslo"

    class Dumpable:
        def model_dump(self):
            return sdk

    assert adapters.guard_gemini(Dumpable(), GEMINI_PARAMS)["days"] == 3


def test_gemini_parallel_function_calls():
    response = candidate(
        [
            gemini_call("get_weather", {"city": "Oslo", "days": 1}),
            gemini_call("get_weather", {"city": "Rome", "days": 2}),
            gemini_call("ping", {}),
        ]
    )
    found = adapters.guard_gemini_calls(response, GEMINI_TOOLS)
    assert [c.arguments.get("city") for c in found] == ["Oslo", "Rome", None]
    second = adapters.guard_gemini(
        response, tools=GEMINI_TOOLS, tool_name="get_weather", index=1
    )
    assert second["city"] == "Rome"
    with pytest.raises(sg.ToolCallNotFoundError, match="index 3"):
        adapters.guard_gemini(response, index=3)


def test_gemini_malformed_function_call_is_rebuilt_from_its_message():
    text = (
        "Malformed function call: "
        'print(default_api.get_weather(city = "Oslo", days = 2))'
        "print(default_api.get_weather("
        'city = \u201cRome\u201d, days = 4, tags = ["a", "b"]))'
    )
    response = candidate([], "MALFORMED_FUNCTION_CALL", text)
    first, second = adapters.guard_gemini_calls(response, GEMINI_TOOLS)
    assert first.arguments == {"city": "Oslo", "days": 2}
    assert second.arguments == {"city": "Rome", "days": 4, "tags": ["a", "b"]}
    assert "recovered_call" in codes(first.repairs)
    picked = adapters.guard_gemini(response, tools=GEMINI_TOOLS, index=1)
    assert picked["city"] == "Rome"
    nested = (
        "default_api.get_weather(city='Oslo', days=2, note=None, "
        "extra={'a': [1, True]})"
    )
    (one,) = adapters.recover_gemini_calls(nested)
    assert one.arguments["extra"] == {"a": [1, True]}
    assert one.name == "get_weather"


@pytest.mark.parametrize(
    "message",
    [
        None,
        "Malformed function call: print(default_api.get_weather('Oslo', 2))",
        "Malformed function call: print(default_api.f(city=os.getcwd()))",
        "Malformed function call: print(default_api.get_weather(city='Oslo'",
        "Malformed function call: nothing callable here",
        "Malformed function call: print(1 +)",
        "Malformed function call: print(1)",
        'Malformed function call: print(handlers[0](city="Oslo"))',
        "Malformed function call: print(default_api.f(city='Oslo",
    ],
)
def test_gemini_unrecoverable_malformed_calls_raise(message):
    response = candidate([], "MALFORMED_FUNCTION_CALL", message)
    with pytest.raises(sg.ToolCallNotFoundError, match="MALFORMED"):
        adapters.guard_gemini(response)


def test_gemini_oversized_or_deeply_nested_call_text_is_refused():
    nested = "default_api.f(a=" + "[" * 60 + "]" * 60 + ")"
    huge = "default_api.f(a='" + "x" * 20_000 + "')"
    for text in (nested, huge):
        response = candidate([], "MALFORMED_FUNCTION_CALL", text)
        with pytest.raises(sg.ToolCallNotFoundError, match="MALFORMED"):
            adapters.guard_gemini(response)
    fine = "default_api.f(a=" + "[" * 10 + "]" * 10 + ")"
    assert len(adapters.recover_gemini_calls(fine)) == 1


def test_gemini_blocked_and_truncated_responses():
    blocked = candidate([], "SAFETY")
    with pytest.raises(sg.RefusalError, match="SAFETY"):
        adapters.guard_gemini(blocked)
    with pytest.raises(sg.RefusalError):
        adapters.guard_gemini_calls(blocked)
    with pytest.raises(sg.ToolCallNotFoundError):
        adapters.guard_gemini(candidate([], "STOP"))
    part = gemini_call("get_weather", {"city": "Oslo", "days": 1})
    cut = candidate([part], "MAX_TOKENS")
    assert adapters.guard_gemini_calls(cut)[0].truncated is True
    with pytest.raises(sg.JSONRepairError, match="cut off"):
        adapters.guard_gemini(cut, allow_truncated=False)


def test_gemini_schema_normalisation():
    raw = {
        "type": "OBJECT",
        "properties": {
            "a": {"type": "STRING"},
            "b": {"type": ["INTEGER", "NULL"]},
            "c": {"type": "TYPE_UNSPECIFIED", "description": "kept"},
            "d": {"type": ["WEIRD"]},
            "e": {"type": 5},
            "f": {"anyOf": [{"type": "NUMBER"}, {"type": "STRING"}]},
            "g": {"type": "ARRAY", "items": {"type": "BOOLEAN"}},
        },
        "required": ["a"],
    }
    clean = adapters.normalize_gemini_schema(raw)
    props = clean["properties"]
    assert clean["type"] == "object" and props["a"] == {"type": "string"}
    assert props["b"]["type"] == ["integer", "null"]
    assert props["c"] == {"description": "kept"} and props["d"] == {}
    assert props["e"] == {}
    assert [s["type"] for s in props["f"]["anyOf"]] == ["number", "string"]
    assert props["g"]["items"] == {"type": "boolean"}
    assert raw["properties"]["a"]["type"] == "STRING"
    assert adapters.normalize_gemini_schema(True) is True


def test_gemini_tool_declarations_in_several_spellings():
    declaration = {"name": "get_weather", "parameters": GEMINI_PARAMS}
    camel = [{"functionDeclarations": [declaration]}]
    part = gemini_call("get_weather", {"city": "Oslo", "days": "2"})
    response = candidate([part])
    assert adapters.guard_gemini(response, tools=camel)["days"] == 2
    strict = {"type": "object", "required": ["city"]}
    json_schema = [{"name": "get_weather", "parametersJsonSchema": strict}]
    empty = candidate([gemini_call("get_weather", {})])
    with pytest.raises(sg.SchemaValidationError):
        adapters.guard_gemini(empty, tools=json_schema)
    loose = {"type": "object"}
    snake = [{"name": "get_weather", "parameters_json_schema": loose}]
    assert adapters.guard_gemini(response, tools=snake)["city"] == "Oslo"
    rogue = candidate([gemini_call("launch_missiles", {})])
    with pytest.raises(sg.ToolCallNotFoundError, match="launch_missiles"):
        adapters.guard_gemini(rogue, tools=GEMINI_TOOLS)
    ping = candidate([gemini_call("ping", {})])
    assert adapters.guard_gemini(ping, tools=GEMINI_TOOLS) == {}


def test_gemini_structured_output_text_ignores_thought_parts():
    parts = [
        {"text": "thinking...", "thought": True},
        {"text": '{"city": "Oslo", '},
        {"text": '"days": "3"}'},
    ]
    got = adapters.guard_gemini(candidate(parts), GEMINI_PARAMS)
    assert got == {"city": "Oslo", "days": 3}
    cut = candidate([{"text": '{"city": "Oslo", "days": 3'}], "MAX_TOKENS")
    assert adapters.guard_gemini_calls(cut)[0].truncated is True


def test_gemini_text_payloads():
    text = '{"city": "Oslo", "days": 3,}'
    assert adapters.guard_gemini(text, GEMINI_PARAMS)["days"] == 3
    call_text = 'print(default_api.get_weather(city="Oslo", days=5))'
    got = adapters.guard_gemini(call_text, GEMINI_PARAMS)
    assert got == {"city": "Oslo", "days": 5}
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_gemini("just words")


@pytest.mark.parametrize(
    "payload",
    [None, 5, {}, {"hello": 1}, {"candidates": []}, {"candidates": [None]}],
)
def test_gemini_unrecognised_payloads_raise(payload):
    with pytest.raises(sg.JSONRepairError):
        adapters.guard_gemini(payload)


# ------------------------------------------------------- across providers
def _openai(args):
    return chat({"tool_calls": [weather_call(json.dumps(args))]})


def _claude(args):
    return reply([tool_use("get_weather", args)])


def _mcp(args):
    return rpc("get_weather", args)


def _gemini(args):
    return candidate([gemini_call("get_weather", args)])


GUARDS = [
    (adapters.guard_openai, _openai),
    (adapters.guard_claude, _claude),
    (adapters.guard_mcp, _mcp),
    (adapters.guard_gemini, _gemini),
]


@pytest.mark.parametrize("guard, build", GUARDS)
def test_every_provider_yields_the_same_clean_dictionary(guard, build):
    messy = {"city": "Oslo", "days": "3", "colour": "blue"}
    assert guard(build(messy), WEATHER) == FORECAST
    with pytest.raises(sg.SchemaValidationError) as info:
        guard(build({"city": "Oslo"}), WEATHER)
    assert info.value.errors[0].pointer == "/days"


@pytest.mark.parametrize("guard, build", GUARDS)
def test_adapters_never_modify_the_payload_or_schema(guard, build):
    payload = build({"city": "Oslo", "days": "3", "colour": "blue"})
    schema = copy.deepcopy(WEATHER)
    snapshot = copy.deepcopy(payload)
    guard(payload, schema)
    assert payload == snapshot and schema == WEATHER


HOSTILE = [
    None, True, 5, 3.5, b"{}", object(), "", "   ", "{", "[", "null",
    "true", "42", '"text"', "[1, 2", '{"a": ', "]]]}}}", [], [None],
    [1, "x"], {}, {"a": 1}, {"choices": None},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": 5}}]},
    {"output": None}, {"output": [None, 5, {"content": 7}]},
    {"content": None}, {"content": [None]},
    {"content": [{"type": "tool_use", "name": 5, "input": []}]},
    {"candidates": None}, {"candidates": [{"content": None}]},
    {"candidates": [{"content": {"parts": "x"}}]},
    {"method": None}, {"method": "tools/call"},
    {"method": "tools/call", "params": {"name": 5, "arguments": 5}},
    {"result": 5}, {"error": 5}, {"message": {"tool_calls": 5}},
    {"type": "tool_use", "input": 5}, {"parts": [None]},
    {"functionCall": 5}, {"function_call": {"name": 5}},
]  # fmt: skip


@pytest.mark.parametrize("guard, build", GUARDS)
def test_hostile_payloads_only_raise_structured_guard_errors(guard, build):
    for payload in HOSTILE:
        try:
            guard(payload, WEATHER)
        except sg.StructuredGuardError:
            continue
        except Exception as error:  # anything else is a bug
            name = type(error).__name__
            raise AssertionError(f"{guard.__name__}({payload!r}): {name}")


def test_new_exceptions_and_exports():
    assert issubclass(sg.ToolCallNotFoundError, sg.StructuredGuardError)
    assert issubclass(sg.ToolCallNotFoundError, LookupError)
    assert issubclass(sg.ToolResultError, sg.StructuredGuardError)
    assert not issubclass(sg.ToolResultError, ValueError)
    required = (
        "guard_openai",
        "guard_claude",
        "guard_mcp",
        "guard_gemini",
        "guard_json",
        "repair_structured_output",
        "StructuredGuardError",
        "JSONRepairError",
        "SchemaValidationError",
    )
    for name in required:
        assert name in sg.__all__ and hasattr(sg, name)
    assert sg.guard_openai is adapters.guard_openai
    plural = {"guard_openai_calls", "guard_claude_calls", "guard_mcp_calls"}
    assert set(adapters.__all__) >= plural | {"guard_gemini_calls"}
    for name in adapters.__all__:
        assert hasattr(adapters, name)
    assert re.fullmatch(r"\d+\.\d+\.\d+", sg.__version__)
