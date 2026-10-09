# structured-guard

[![tests](https://github.com/dev-stratum/structured-guard/actions/workflows/tests.yml/badge.svg)](https://github.com/dev-stratum/structured-guard/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/structured-guard)](https://pypi.org/project/structured-guard/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

**A zero-dependency reliability shield for LLM structured outputs, tool calling and the
Model Context Protocol.** One small library repairs, validates and explains the malformed,
truncated or mismatched JSON that OpenAI, Claude, Gemini, MCP servers and local models
(Ollama, vLLM) hand back, using nothing but the Python standard library. It imports no
provider SDK, makes no network calls, and works on plain dictionaries and strings.

```python
from structured_guard import guard_claude, guard_gemini, guard_mcp, guard_openai
```

Four dialects in, one clean dictionary out, each checked against your JSON Schema.

## Why tool calling breaks, on every provider

Tool arguments and structured outputs are *generated text*. When generation is cut off, is
not schema-constrained, or crosses a process boundary, the same few things go wrong:

- **OpenAI.** Structured Outputs guarantee a schema for responses that complete in strict
  mode. The official guide lists the exceptions: refusals and responses that stop at the
  token limit. Outside strict mode (JSON mode, non-strict function calls, older models and
  OpenAI-compatible servers) nothing constrains the arguments.
- **Claude.** A response that stops at `max_tokens` can end in an incomplete `tool_use`
  block, and a tool input is only as good as the model's adherence to your `input_schema`.
  Values that should be arrays or objects occasionally arrive as JSON-encoded strings.
- **MCP.** `tools/call` arguments and `CallToolResult` payloads travel as JSON-RPC over stdio
  or HTTP. A truncated or noisy transport line, double-encoded `arguments`, or a result whose
  `structuredContent` does not match the declared `outputSchema` should not crash an agent,
  and a tool that reports `isError` is not the same as malformed JSON.
- **Gemini.** A response can finish with `MALFORMED_FUNCTION_CALL`; developers report that the
  attempted call is visible in the finish message as Python-style text. Integers arrive as
  floats, and function schemas use upper-case type names such as `OBJECT`.
- **Local models.** Models served by Ollama or vLLM often print a tool call as text (a JSON
  object with `name` and `arguments` or `parameters`, a list of them, or objects wrapped in
  `tool_call` tags), and may be cut off mid-object.

Every failure costs a crashed run or a paid retry. `structured-guard` makes the recoverable
cases recoverable and the unrecoverable cases explicit, typed and cheap to correct.

## Install

```bash
pip install structured-guard
```

or straight from GitHub:

```bash
pip install git+https://github.com/dev-stratum/structured-guard.git
```

or from a clone:

```bash
git clone https://github.com/dev-stratum/structured-guard.git
cd structured-guard
pip install .
```

Requires Python 3.10 or newer. There are no runtime dependencies. `pip install -e ".[dev]"`
adds pytest, the only development dependency.

## Quickstart

Every example below runs as written; the payloads are plain dictionaries and strings.

### OpenAI: function calling and Structured Outputs

```python
from structured_guard import guard_openai

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "minimum": 1},
    },
    "required": ["city", "days"],
    "additionalProperties": False,
}

response = {  # a Chat Completions response that stopped at the token limit
    "choices": [
        {
            "finish_reason": "length",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_forecast",
                            "arguments": '{"city": "Oslo", "days": "3',
                        },
                    }
                ],
            },
        }
    ]
}

print(guard_openai(response, schema))
# {'city': 'Oslo', 'days': 3}
```

Pass the SDK response object itself if you prefer (anything with `model_dump()` or
`to_dict()`), or a Responses API response. A message with JSON text instead of tool calls is
treated as Structured Output, and a refusal raises `RefusalError`.

### Claude: `tool_use` blocks

```python
from structured_guard import guard_claude

trip_schema = {
    "type": "object",
    "properties": {
        "stops": {"type": "array", "items": {"type": "string"}},
        "days": {"type": "integer"},
    },
    "required": ["stops"],
}

response = {
    "type": "message",
    "role": "assistant",
    "stop_reason": "tool_use",
    "content": [
        {"type": "text", "text": "Let me plan that."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "plan_trip",
            "input": {"stops": '["Oslo", "Bergen"]', "days": "3"},
        },
    ],
}

print(guard_claude(response, trip_schema))
# {'stops': ['Oslo', 'Bergen'], 'days': 3}
```

A `max_tokens` stop that ends inside a `tool_use` block marks that call as truncated, and
`stop_reason: "refusal"` raises `RefusalError`.

### MCP: `tools/call` requests and `CallToolResult`

```python
from structured_guard import guard_mcp

tools = [
    {
        "name": "read_file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {"lines": {"type": "integer"}},
            "required": ["lines"],
        },
    }
]

# A transport line that was cut off in the middle of the message.
line = (
    '{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": '
    '{"name": "read_file", "arguments": {"path": "./notes.txt", "max_bytes": "512"'
)
print(guard_mcp(line, tools=tools))
# {'path': './notes.txt', 'max_bytes': 512}

result = {"structuredContent": {"lines": "42"}, "isError": False}
print(guard_mcp(result, tools=tools, tool_name="read_file"))
# {'lines': 42}
```

A result with `isError` set raises `ToolResultError`, a result without `structuredContent`
falls back to the JSON in its text content, and a call to a tool that `tools` does not define
raises `ToolCallNotFoundError`.

### Gemini: `function_call` parts

```python
from structured_guard import guard_gemini

gemini_schema = {
    "type": "OBJECT",
    "properties": {"city": {"type": "STRING"}, "days": {"type": "INTEGER"}},
    "required": ["city", "days"],
}

response = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_forecast",
                            "args": {"city": "Oslo", "days": 3.0},
                        }
                    }
                ],
            },
            "finishReason": "STOP",
        }
    ]
}

print(guard_gemini(response, gemini_schema))
# {'city': 'Oslo', 'days': 3}
```

Upper-case type names are normalised, and both the REST spelling (`functionCall`) and the
SDK spelling (`function_call`) work. A `MALFORMED_FUNCTION_CALL` can often be rebuilt from its
message (the text is parsed with `ast` and never executed):

```python
from structured_guard.adapters import guard_gemini_calls

blocked = {
    "candidates": [
        {
            "content": {"parts": []},
            "finishReason": "MALFORMED_FUNCTION_CALL",
            "finishMessage": 'Malformed function call: print(default_api.get_forecast(city = "Oslo", days = 2))',
        }
    ]
}
print([call.arguments for call in guard_gemini_calls(blocked)])
# [{'city': 'Oslo', 'days': 2}]
```

### Local models: Ollama and vLLM

vLLM and other OpenAI-compatible servers return the OpenAI shape, so `guard_openai` already
works. Ollama's native chat response works too, and when a model prints its tool call as text,
pass `tools=` (or `tool_name=`) so the text is read as a call:

```python
from structured_guard import guard_openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city", "days"],
            },
        },
    }
]

reply = {  # what Ollama's chat endpoint returned
    "model": "local-model",
    "message": {
        "role": "assistant",
        "content": 'Sure, checking now.\n{"name": "get_forecast", "parameters": {"city": "Oslo", "days": "2"}}',
    },
    "done": True,
}

print(guard_openai(reply, tools=tools))
# {'city': 'Oslo', 'days': 2}
```

### Parallel calls and plain text

Every `guard_*` function has a `guard_*_calls` twin in `structured_guard.adapters` that
returns every call as a `GuardedCall` (`name`, `call_id`, `arguments`, `repairs`,
`truncated`), each checked against its own tool. To pick one call instead, use `tool_name=`,
`call_id=` or `index=`.

```python
from structured_guard import repair_structured_output

print(repair_structured_output("{'ok': True, 'items': [1, 2, 3,],} Hope that helps!"))
# {'ok': True, 'items': [1, 2, 3]}
```

When validation fails, the exception says exactly what to fix:

```python
from structured_guard import SchemaValidationError, guard_claude

response = {"content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]}
schema = {"type": "object", "properties": {"stops": {"type": "array"}}, "required": ["stops"]}

try:
    guard_claude(response, schema)
except SchemaValidationError as error:
    print("; ".join(str(issue) for issue in error.errors))
# $.stops: missing required property 'stops'
```

## How it works

```
 provider payload (dict, SDK object, text)
        |
        v
  adapter          finds the calls, refusals and "stopped early" signals
        |
        v
  sanitizer        strips fences, filler and call tags
        |
        v
  parser           json.loads fast path, else a single-pass stack-based repair
        |
        v
  validator        lossless coercion, defaults, JSON Schema checks with paths
        |
        v
  clean dict   or   a typed exception that says what is wrong
```

| Module | Responsibility |
| --- | --- |
| `adapters/` | One adapter per dialect: `openai`, `claude`, `mcp`, `gemini`, plus shared plumbing. |
| `sanitizer.py` | Extracts the JSON from fences and chatter; reads OpenAI-style responses. |
| `parser.py` | Stack-based lenient parser; nesting is limited by `max_depth`, not by Python's recursion limit. |
| `validator.py` | Standard-library JSON Schema subset: validation plus conservative coercion. |
| `core.py` | `repair_structured_output`, `inspect_structured_output`, `guard_json`, `GuardResult`. |

## What gets repaired

| Problem | Handling |
| --- | --- |
| Markdown fences, greetings, trailing commentary, `tool_call` tags | Removed |
| Unclosed objects and arrays, cut-off strings, numbers, `true`/`false`/`null` | Closed and reconstructed, flagged as truncated |
| Single quotes, unquoted keys, trailing or missing commas, comments | Normalised |
| `True`, `False`, `None`, `NaN`, `Infinity` | Converted |
| Unescaped quotes and raw control characters inside strings | Accepted |
| Arguments encoded as JSON twice, arrays or objects sent as strings | Decoded |
| Empty arguments for a tool without parameters | An empty object |
| `"3"` for an integer, `3.0` for an integer, one value for an array | Coerced losslessly |
| Properties the schema forbids, `null` for optional properties | Dropped |
| Calls to tools you never defined | `ToolCallNotFoundError` |
| Refusals, tool errors, protocol errors | `RefusalError`, `ToolResultError` |

Every change is recorded with its JSON path in `GuardedCall.repairs`. Reconstructing cut-off
data is a guess, so results carry a `truncated` flag and `allow_truncated=False` turns any
such guess into an error.

## structured-guard vs naive approaches

| Capability | `json.loads` | Hand-written per-provider parsing | structured-guard |
| --- | :---: | :---: | :---: |
| Reads OpenAI, Claude, MCP, Gemini and Ollama payload shapes | no | one provider at a time | yes |
| Fences, filler and call tags around the JSON | no | partly | yes |
| Truncated arguments | raises | raises, or silently wrong | reconstructed and flagged |
| Arguments encoded twice, arrays sent as strings | no | rarely | yes |
| Schema validation with JSON paths | no | needs a dependency | yes, standard library |
| Calls to tools you did not define | no | by hand | yes |
| Gemini `MALFORMED_FUNCTION_CALL` | crash | crash | rebuilt when possible |
| MCP `isError` versus protocol errors | n/a | by hand | separate exceptions |
| Refusals | n/a | by hand | `RefusalError` |
| Provider SDK required | no | often | never |
| Runtime dependencies | 0 | often several | 0 |

## API

| Name | Purpose |
| --- | --- |
| `guard_openai`, `guard_claude`, `guard_mcp`, `guard_gemini` | Repair and validate one tool call (or one structured output); return a dict. |
| `structured_guard.adapters.guard_*_calls` | The same for every call in the payload; return `GuardedCall` objects. |
| `repair_structured_output(source, schema=None)` | Repair and validate any model text or OpenAI-style response. |
| `inspect_structured_output(source, schema=None)` | The same, but returns a `GuardResult` with `retry_prompt()` instead of raising. |
| `guard_json(schema=...)` | Decorator for sync and async functions that return model output. |

Common keyword options: `tools` (tool definitions, used to look up each call's schema),
`tool_name`, `call_id`, `index`, and `coerce`, `apply_defaults`, `drop_extra`,
`allow_truncated`, `max_depth`, `max_issues`. Unknown options raise `TypeError`.

| Exception | Raised when |
| --- | --- |
| `StructuredGuardError` | Base class of everything below. |
| `JSONRepairError` | No JSON can be recovered, or truncation is not allowed. |
| `SchemaValidationError` | The data violates the schema; has `.errors` and the repaired `.value`. |
| `RefusalError` | The model refused instead of answering. |
| `ToolCallNotFoundError` | No (defined) tool call matches. |
| `ToolResultError` | An MCP result has `isError`, or the response is a JSON-RPC error. |
| `InvalidSchemaError` | The schema itself is malformed or unsupported. |

## Schema support

Supported: `type` (including lists and `nullable`), `enum`, `const`, `properties`, `required`,
`additionalProperties`, `items`, `minItems`, `maxItems`, `minLength`, `maxLength`, `pattern`,
`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `allOf`, `anyOf`,
`oneOf` (treated like `anyOf`), local `$ref` and `$defs`, and `default`. Not enforced:
`format`, `patternProperties`, `prefixItems`, `if`/`then`/`else`, `not`, `contains`,
`uniqueItems`, `minProperties`, `maxProperties` and `unevaluated*`;
`structured_guard.validator.unsupported_keywords(schema)` lists the ones a schema uses.

## Development

```bash
python -m pytest
```

The tests use synthetic payloads only: no network access, no API keys, no provider SDKs. The
suite also checks the repository itself: only standard library imports, 79-column lines
without tabs or trailing whitespace, no absolute paths, usernames or placeholder markers, and
package metadata that matches the project.

## Limitations

- JSON Schema support is a deliberate subset; this is not a full validator.
- Repair is heuristic by nature. Check `truncated` and `repairs` when the data drives
  something important.
- Provider formats change. The adapters read documented fields and ignore unknown ones; if a
  payload shape is not recognised you get a `JSONRepairError`, never a silent guess.

## Contributing, security and changelog

Bug reports and "payload that broke" reports are welcome; see
[CONTRIBUTING.md](https://github.com/dev-stratum/structured-guard/blob/main/CONTRIBUTING.md).
Please report vulnerabilities privately as described in
[SECURITY.md](https://github.com/dev-stratum/structured-guard/blob/main/SECURITY.md).
The release history is in
[CHANGELOG.md](https://github.com/dev-stratum/structured-guard/blob/main/CHANGELOG.md).

## License

MIT. See [LICENSE](https://github.com/dev-stratum/structured-guard/blob/main/LICENSE).

## Trademark notice

structured-guard is an independent open-source project. It is not affiliated with, endorsed
by, or sponsored by OpenAI, Anthropic, Google or the Model Context Protocol project. Product
and company names are trademarks of their respective owners and are used here only to
describe compatibility.
