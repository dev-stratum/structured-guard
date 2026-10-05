"""Find the JSON inside raw model output.

Language models wrap JSON in markdown fences, greetings, apologies and
trailing commentary. The helpers here locate where the JSON starts so the
parser never has to deal with the surrounding filler. They also read the text
(or tool-call arguments) out of OpenAI-style response objects.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .exceptions import JSONRepairError

__all__ = [
    "QUOTES",
    "ModelText",
    "Payload",
    "as_mapping",
    "balanced_end",
    "check_prefer",
    "extract_payloads",
    "read_response",
    "strip_to_json",
    "text_of",
]

QUOTES = {'"': '"', "'": "'", "\u201c": "\u201d"}

_MAX_CANDIDATES = 16
_OPEN = re.compile(r"[\[{]")
_FENCE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n?(.*?)(?:```|\Z)", re.DOTALL
)
_JSON_LANGS = frozenset({"", "json", "jsonc", "json5"})
_Output = tuple[str, str, "str | None"]


@dataclass(frozen=True)
class Payload:
    """A place in the text where a JSON value may start."""

    text: str
    start: int
    base: int
    fenced: bool


@dataclass(frozen=True)
class ModelText:
    """Text read from a model response."""

    text: str
    refusal: str | None
    truncated: bool


def check_prefer(prefer: str | None) -> None:
    """Raise ``ValueError`` unless *prefer* is None, 'object' or 'array'."""
    if prefer not in (None, "object", "array"):
        raise ValueError("prefer must be None, 'object' or 'array'")


def balanced_end(text: str, start: int) -> int:
    """Index of the bracket that closes the one at *start*.

    Returns ``len(text)`` when the bracket is never closed. Brackets inside
    quoted strings are ignored.
    """
    depth, i, n = 0, start, len(text)
    while i < n:
        ch = text[i]
        if ch in QUOTES:
            closer = QUOTES[ch]
            i += 1
            while i < n and text[i] != closer:
                i += 2 if text[i] == "\\" else 1
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


def _documents(text: str) -> list[tuple[str, int, bool]]:
    """Text fragments worth searching, most specific first."""
    fenced = [
        (match.group(2), match.start(2), True)
        for match in _FENCE.finditer(text)
        if match.group(1).lower() in _JSON_LANGS
    ][:_MAX_CANDIDATES]

    def mask(match: re.Match[str]) -> str:
        # Blank out fences for other languages so their brackets are ignored.
        if match.group(1).lower() in _JSON_LANGS:
            return match.group()
        return " " * len(match.group())

    return [*fenced, (_FENCE.sub(mask, text), 0, False)]


def extract_payloads(text: str, prefer: str | None = None) -> list[Payload]:
    """List the places in *text* where a JSON object or array may start.

    Fenced ``json`` blocks come first, then the whole text with fences for
    other languages blanked out. Within each fragment the brackets of the
    *prefer* kind ('object' or 'array') are tried first.
    """
    check_prefer(prefer)
    payloads: list[Payload] = []
    for doc, base, fenced in _documents(text):
        starts = [match.start() for match in _OPEN.finditer(doc)]
        if prefer is not None:
            wanted = "{" if prefer == "object" else "["
            starts.sort(key=lambda pos, doc=doc: doc[pos] != wanted)
        payloads.extend(
            Payload(doc, start, base, fenced)
            for start in starts[:_MAX_CANDIDATES]
        )
    return payloads


def strip_to_json(text: str, prefer: str | None = None) -> str:
    """Return *text* with fences, filler and commentary removed.

    The result runs from the first opening bracket to its matching closing
    bracket (or to the end of the text if the bracket is never closed).

    Raises:
        JSONRepairError: If the text contains no object or array.
    """
    payloads = extract_payloads(text, prefer)
    if not payloads:
        raise JSONRepairError("no JSON object or array found in the text")
    first = payloads[0]
    end = balanced_end(first.text, first.start)
    stop = end + 1
    return first.text[first.start:stop]


# -- reading OpenAI-style response objects --------------------------------
def as_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    for method in ("model_dump", "to_dict"):
        convert = getattr(response, method, None)
        if callable(convert):
            data = convert()
            if isinstance(data, Mapping):
                return data
    kind = type(response).__name__
    raise JSONRepairError(f"cannot read model output from a {kind} object")


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _arguments_of(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, (Mapping, list)):
        return json.dumps(arguments)
    return ""


def _read_chat(
    data: Mapping[str, Any],
) -> tuple[list[_Output], str | None, bool]:
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
    outputs: list[_Output] = []
    text = text_of(message.get("content"))
    if text.strip():
        outputs.append(("text", text, None))
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            arguments = _arguments_of(function.get("arguments"))
            outputs.append(("tool_call", arguments, function.get("name")))
    refusal = message.get("refusal")
    refusal = refusal if isinstance(refusal, str) and refusal else None
    return outputs, refusal, choice.get("finish_reason") == "length"


def _read_responses(
    data: Mapping[str, Any],
) -> tuple[list[_Output], str | None, bool]:
    items = data["output"]
    if not isinstance(items, list):
        raise JSONRepairError("'output' must be a list")
    outputs: list[_Output] = []
    refusal: str | None = None
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "function_call":
            arguments = _arguments_of(item.get("arguments"))
            outputs.append(("tool_call", arguments, item.get("name")))
        elif item.get("type") == "message" and isinstance(
            item.get("content"), list
        ):
            for part in item["content"]:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if part.get("type") == "refusal" and isinstance(
                    part.get("refusal"), str
                ):
                    refusal = part["refusal"]
                elif (
                    part.get("type") == "output_text"
                    and isinstance(text, str)
                    and text.strip()
                ):
                    outputs.append(("text", text, None))
    details = data.get("incomplete_details")
    truncated = (
        data.get("status") == "incomplete"
        and isinstance(details, Mapping)
        and details.get("reason") == "max_output_tokens"
    )
    return outputs, refusal, truncated


def _select(outputs: list[_Output], tool_name: str | None) -> str:
    if tool_name is not None:
        for kind, text, name in outputs:
            if kind == "tool_call" and name == tool_name:
                return text
        raise JSONRepairError(
            f"the response has no tool call named {tool_name!r}"
        )
    texts = [text for kind, text, _ in outputs if kind == "text"]
    pool = texts or [text for _, text, _ in outputs]
    if not pool:
        raise JSONRepairError("the response has neither text nor a tool call")
    return pool[0]


def read_response(source: Any, tool_name: str | None = None) -> ModelText:
    """Read the model text from a string or an OpenAI-style response.

    Chat Completions and Responses API objects are understood, as plain
    dictionaries or as SDK objects that provide ``model_dump()`` or
    ``to_dict()``. With *tool_name*, the arguments of that tool call are
    returned; otherwise message text wins over tool calls.

    Raises:
        JSONRepairError: If no text can be read from *source*.
    """
    if isinstance(source, str):
        return ModelText(source, None, False)
    data = as_mapping(source)
    if "choices" in data:
        outputs, refusal, truncated = _read_chat(data)
    elif "output" in data:
        outputs, refusal, truncated = _read_responses(data)
    else:
        raise JSONRepairError(
            "expected a Chat Completions or Responses API response"
        )
    if refusal is not None:
        return ModelText("", refusal, truncated)
    return ModelText(_select(outputs, tool_name), None, truncated)
