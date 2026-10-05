"""Stack-based, lenient JSON parser.

The parser walks the text once and keeps an explicit stack of open
containers instead of recursing, so nesting depth is limited only by
``max_depth`` and never by Python's recursion limit. It repairs as it goes
and records every change as a :class:`~structured_guard.models.Repair`.

Truncated input is reconstructed: unterminated strings are closed, open
objects and arrays are auto-closed, and cut-off ``true``/``false``/``null``
and numbers are completed. A key that has no value yet is dropped.
Reconstruction cannot know what was lost, so the result reports
``truncated=True`` whenever it had to guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, NoReturn

from .exceptions import JSONRepairError
from .models import DEFAULT_MAX_DEPTH, Repair
from .sanitizer import (
    QUOTES,
    Payload,
    balanced_end,
    check_prefer,
    extract_payloads,
)

__all__ = ["ParseResult", "parse_json", "repair_json_text"]

_MISSING: Any = object()
_MAX_NUMBER_CHARS = 4000
_INF = float("inf")

_CHUNK = {
    opener: re.compile("[^" + re.escape(closer) + r"\\]+")
    for opener, closer in QUOTES.items()
}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]*)?")
_JSON_NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)
_DANGLING_EXPONENT = re.compile(r"[eE][+-]?$")
_SIGN_TAIL = re.compile(r"[+-]?\.?\s*")
_WORD = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_KEY = re.compile(r"[^\W\d][\w$-]*")
_HEX_ONLY = re.compile(r"[0-9a-fA-F]*")
_SURROGATE = re.compile("[\ud800-\udfff]")
_CONTROL = re.compile(r"[\x00-\x1f]")
_LITERALS = {
    "true": True,
    "false": False,
    "null": None,
    "True": True,
    "False": False,
    "None": None,
}
_CANONICAL = frozenset({"true", "false", "null"})
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_FOLLOW = ",:}]/"
_S_VALUE, _S_KEY, _S_COLON, _S_AFTER, _S_DONE = range(5)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing one JSON value."""

    value: Any
    repairs: tuple[Repair, ...]
    truncated: bool
    end: int


class _Frame:
    """One open object or array on the parser stack."""

    __slots__ = ("items", "is_dict", "key", "after_comma")

    def __init__(self, is_dict: bool) -> None:
        self.items: Any = {} if is_dict else []
        self.is_dict = is_dict
        self.key = ""
        self.after_comma = False


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def _finite_float(token: str) -> float:
    number = float(token)
    if number in (_INF, -_INF):
        raise ValueError(token)
    return number


def _join(parts: list[str]) -> str:
    text = "".join(parts)
    if _SURROGATE.search(text):
        # Pair up \ud83d\ude00 style escapes and replace lone surrogates.
        data = text.encode("utf-16", "surrogatepass")
        return data.decode("utf-16", "replace")
    return text


def _clean_number(token: str) -> str:
    token = _DANGLING_EXPONENT.sub("", token)
    return token[:-1] if token.endswith(".") else token


def _starts_item(is_dict: bool, ch: str) -> bool:
    if is_dict:
        return ch in QUOTES or ch == "_" or ch.isalpha()
    return ch in "{[+-." or ch in QUOTES or ch.isalnum()


class _Parser:
    """Single-use parser for one candidate payload."""

    def __init__(
        self,
        text: str,
        start: int,
        base: int,
        max_depth: int,
        heuristic: bool,
    ) -> None:
        self.s = text
        self.n = len(text)
        self.i = start
        self.base = base
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.repairs: list[Repair] = []
        self.truncated = False
        self.path: list[str | int] = []

    # -- helpers ---------------------------------------------------------
    def _note(self, code: str, message: str) -> None:
        self.repairs.append(Repair(code, message, tuple(self.path)))

    def _fail(self, message: str) -> NoReturn:
        raise JSONRepairError(message, offset=self.base + self.i)

    def _truncate(self, message: str) -> None:
        if not self.truncated:
            self.truncated = True
            self._note("truncated", message)

    def _skip(self) -> None:
        s, n = self.s, self.n
        while self.i < n:
            ch = s[self.i]
            if ch.isspace() or ch == "\ufeff":
                self.i += 1
            elif s.startswith("//", self.i):
                end = s.find("\n", self.i)
                self.i = n if end == -1 else end + 1
                self._note("comment", "removed a // comment")
            elif s.startswith("/*", self.i):
                end = s.find("*/", self.i + 2)
                self.i = n if end == -1 else end + 2
                self._note("comment", "removed a /* */ comment")
            else:
                return

    # -- state machine -----------------------------------------------------
    def run(self, complete: bool) -> Any:
        """Parse one value; with *complete*, nothing may follow it."""
        stack: list[_Frame] = []
        state = _S_VALUE
        root: Any = _MISSING
        while state != _S_DONE:
            self._skip()
            if self.i >= self.n:
                return self._finish(stack)
            ch = self.s[self.i]
            if state == _S_VALUE:
                state, root = self._on_value(stack, root, ch)
            elif state == _S_KEY:
                state, root = self._on_key(stack, root, ch)
            elif state == _S_COLON:
                if ch != ":":
                    self._fail("expected ':' after an object key")
                self.i += 1
                state = _S_VALUE
            else:
                state, root = self._on_after(stack, root, ch)
        if complete:
            self._skip()
            if self.i < self.n:
                self._fail("unexpected characters after the value")
        return root

    def _finish(self, stack: list[_Frame]) -> Any:
        """Handle the end of the input: auto-close everything still open."""
        if not stack:
            self._fail("no JSON value found")
        self._truncate(
            "the input ended early; open brackets were closed and the "
            "last value may be incomplete"
        )
        value = stack[-1].items
        for parent in reversed(stack[:-1]):
            if parent.is_dict:
                parent.items[parent.key] = value
            else:
                parent.items.append(value)
            value = parent.items
        return value

    def _on_value(
        self, stack: list[_Frame], root: Any, ch: str
    ) -> tuple[int, Any]:
        top = stack[-1] if stack else None
        if top is not None and not top.is_dict and ch in "],":
            if ch == ",":
                self._stray_comma()
                return _S_VALUE, root
            return self._close(stack, root)
        self._begin(stack)
        if ch in "{[":
            self._push(stack, ch == "{")
            return (_S_KEY if ch == "{" else _S_VALUE), root
        if ch in QUOTES:
            value = self._string()
        else:
            value = self._scalar(len(stack))
            if value is _MISSING:
                self._end(stack)
                self.i = self.n
                return _S_VALUE, root
        return self._attach(stack, root, value)

    def _on_key(
        self, stack: list[_Frame], root: Any, ch: str
    ) -> tuple[int, Any]:
        if ch == "}":
            return self._close(stack, root)
        if ch == ",":
            self._stray_comma()
            return _S_KEY, root
        stack[-1].key = self._key()
        return _S_COLON, root

    def _on_after(
        self, stack: list[_Frame], root: Any, ch: str
    ) -> tuple[int, Any]:
        top = stack[-1]
        if ch == ",":
            self.i += 1
            top.after_comma = True
            return (_S_KEY if top.is_dict else _S_VALUE), root
        if ch == ("}" if top.is_dict else "]"):
            return self._close(stack, root)
        if _starts_item(top.is_dict, ch):
            self._note("missing_comma", "inserted a missing comma")
            return (_S_KEY if top.is_dict else _S_VALUE), root
        self._fail("expected ',' or a closing bracket")

    def _stray_comma(self) -> None:
        self.i += 1
        self._note("extra_comma", "removed a stray comma")

    def _begin(self, stack: list[_Frame]) -> None:
        """Extend the location path as a new value starts."""
        if stack:
            top = stack[-1]
            self.path.append(top.key if top.is_dict else len(top.items))

    def _end(self, stack: list[_Frame]) -> None:
        if stack:
            self.path.pop()

    def _push(self, stack: list[_Frame], is_dict: bool) -> None:
        if len(stack) >= self.max_depth:
            self._fail(f"nesting is deeper than {self.max_depth} levels")
        stack.append(_Frame(is_dict))
        self.i += 1

    def _close(self, stack: list[_Frame], root: Any) -> tuple[int, Any]:
        frame = stack.pop()
        self.i += 1
        if frame.after_comma:
            self._note("trailing_comma", "removed a trailing comma")
        return self._attach(stack, root, frame.items)

    def _attach(
        self, stack: list[_Frame], root: Any, value: Any
    ) -> tuple[int, Any]:
        """Store a finished value in its parent (or make it the root)."""
        if not stack:
            return _S_DONE, value
        parent = stack[-1]
        if parent.is_dict:
            parent.items[parent.key] = value
        else:
            parent.items.append(value)
        parent.after_comma = False
        self.path.pop()
        return _S_AFTER, root

    # -- tokens ------------------------------------------------------------
    def _key(self) -> str:
        if self.s[self.i] in QUOTES:
            return self._string()
        match = _KEY.match(self.s, self.i)
        if match is None:
            self._fail("expected an object key")
        self.i = match.end()
        self._note("unquoted_key", f"quoted the bare key {match.group()!r}")
        return match.group()

    def _string(self) -> str:
        s, n = self.s, self.n
        opener = s[self.i]
        closer = QUOTES[opener]
        chunk = _CHUNK[opener]
        if opener != '"':
            self._note(
                "quotes", "replaced a non-standard quote with a double quote"
            )
        self.i += 1
        parts: list[str] = []
        raw_control = False
        escaped_quote = False
        while True:
            match = chunk.match(s, self.i)
            if match:
                text = match.group()
                raw_control = raw_control or bool(_CONTROL.search(text))
                parts.append(text)
                self.i = match.end()
            if self.i >= n:
                if escaped_quote:
                    self._fail("unterminated string")
                self._truncate(
                    "the input ended inside a string; it was closed"
                )
                break
            if s[self.i] == closer:
                if self.heuristic and not self._closes_string():
                    parts.append(closer)
                    self.i += 1
                    escaped_quote = True
                    continue
                self.i += 1
                break
            self._escape(parts)
        if raw_control:
            self._note(
                "control_characters",
                "accepted raw control characters inside a string",
            )
        if escaped_quote:
            self._note(
                "unescaped_quote", "kept unescaped quotes inside a string"
            )
        return _join(parts)

    def _closes_string(self) -> bool:
        """Does the quote at the cursor end the string (heuristic mode)?"""
        s, n = self.s, self.n
        j = self.i + 1
        while j < n and s[j].isspace():
            j += 1
        return j >= n or s[j] in _FOLLOW

    def _escape(self, parts: list[str]) -> None:
        s, n = self.s, self.n
        if self.i + 1 >= n:
            self.i = n  # a dangling backslash at the very end is dropped
            return
        nxt = s[self.i + 1]
        if nxt in _ESCAPES:
            parts.append(_ESCAPES[nxt])
            self.i += 2
        elif nxt == "u":
            first = self.i + 2
            digits = s[first:first + 4]
            if len(digits) == 4 and _HEX_ONLY.fullmatch(digits):
                parts.append(chr(int(digits, 16)))
                self.i += 6
            elif len(digits) < 4 and _HEX_ONLY.fullmatch(digits):
                self.i = n  # the escape itself was cut off
            else:
                self._note("invalid_escape", "kept an invalid \\u escape")
                parts.append("\\u")
                self.i += 2
        elif nxt == "'":
            self._note("invalid_escape", "unescaped \\' to an apostrophe")
            parts.append("'")
            self.i += 2
        else:
            self._note("invalid_escape", f"kept the invalid escape \\{nxt}")
            parts.append("\\" + nxt)
            self.i += 2

    def _scalar(self, depth: int) -> Any:
        s, n, i = self.s, self.n, self.i
        match = _NUMBER.match(s, i)
        if match:
            return self._number(match)
        if s[i] in "+-" and s.startswith("Infinity", i + 1):
            self.i = i + 9
            self._note("non_finite", "replaced a non-finite number with null")
            return None
        match = _WORD.match(s, i)
        if match is None:
            if depth and _SIGN_TAIL.fullmatch(s, i):
                return _MISSING
            self._fail(f"unexpected character {s[i]!r}")
        word, end = match.group(), match.end()
        self.i = end
        if word in _LITERALS:
            if word not in _CANONICAL:
                self._note("literal", f"converted {word} to its JSON form")
            return _LITERALS[word]
        if word in ("NaN", "Infinity"):
            self._note("non_finite", "replaced a non-finite number with null")
            return None
        if depth and end == n:
            for literal, value in _LITERALS.items():
                if literal.startswith(word):
                    self._note(
                        "reconstructed_literal",
                        f"completed the cut-off literal {word!r} to {literal}",
                    )
                    return value
        self.i = i
        self._fail(f"unexpected token {word!r}")

    def _number(self, match: re.Match[str]) -> Any:
        token = match.group()
        self.i = match.end()
        if len(token) > _MAX_NUMBER_CHARS:
            self._fail("number literal is too long")
        if _JSON_NUMBER.fullmatch(token) is None:
            token = _clean_number(token)
            self._note("number", "normalised a non-standard number")
        if any(c in token for c in ".eE"):
            number = float(token)
            if number in (_INF, -_INF):
                self._fail("number is out of range")
            return number
        return int(token)


def parse_json(
    text: str,
    start: int = 0,
    *,
    base: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
    complete: bool = False,
) -> ParseResult:
    """Leniently parse the JSON value that begins at ``text[start]``.

    A strict pass runs first. If it fails, a second pass additionally
    treats quotes that are not followed by a delimiter as part of the string
    (for example ``"He said "hi" to me"``). The first error is raised if both
    passes fail.

    Args:
        text: The text to parse.
        start: Index where the value begins.
        base: Offset of *text* inside the original input, for error offsets.
        max_depth: Maximum nesting of objects and arrays.
        complete: If true, only whitespace and comments may follow the value.

    Raises:
        JSONRepairError: If no value can be recovered.
    """
    parser = _Parser(text, start, base, max_depth, heuristic=False)
    try:
        value = parser.run(complete)
    except JSONRepairError as error:
        parser = _Parser(text, start, base, max_depth, heuristic=True)
        try:
            value = parser.run(complete)
        except JSONRepairError:
            raise error from None
    repairs = tuple(parser.repairs)
    return ParseResult(value, repairs, parser.truncated, parser.i)


def _decorate(result: ParseResult, payload: Payload) -> ParseResult:
    """Add notes about what surrounded the JSON inside the payload."""
    doc = payload.text
    repairs: list[Repair] = []
    if payload.fenced:
        repairs.append(
            Repair("code_fence", "extracted the JSON from a markdown fence")
        )
    before = doc[:payload.start].replace("\ufeff", "")
    if before.strip():
        repairs.append(Repair("leading_text", "ignored text before the JSON"))
    repairs.extend(result.repairs)
    after = doc[result.end:]
    if after.strip():
        repairs.append(Repair("trailing_text", "ignored text after the JSON"))
    return ParseResult(
        result.value,
        tuple(repairs),
        result.truncated,
        payload.base + result.end,
    )


def repair_json_text(
    text: str,
    *,
    prefer: str | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ParseResult:
    """Turn raw model output into Python data.

    Valid JSON takes a :func:`json.loads` fast path. Anything else is
    cleaned up by the sanitizer and parsed leniently, trying each place a
    JSON value could start until one parses.

    Args:
        text: Raw model output, possibly wrapped in prose or a code fence.
        prefer: 'object' or 'array' to try that kind of value first.
        max_depth: Maximum nesting accepted by the lenient parser.

    Raises:
        JSONRepairError: If no JSON value can be recovered.
        TypeError: If *text* is not a ``str``.
        ValueError: If *prefer* or *max_depth* is invalid.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    check_prefer(prefer)
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    try:
        value = json.loads(
            text, parse_constant=_reject_constant, parse_float=_finite_float
        )
        return ParseResult(value, (), False, len(text))
    except (ValueError, RecursionError):
        pass

    first_error: JSONRepairError | None = None
    failed: list[tuple[int, int, int]] = []
    for payload in extract_payloads(text, prefer):
        doc, start = payload.text, payload.start
        if any(k == id(doc) and lo <= start <= hi for k, lo, hi in failed):
            continue  # inside a region that already failed to parse
        try:
            result = parse_json(
                doc, start, base=payload.base, max_depth=max_depth
            )
        except JSONRepairError as error:
            failed.append((id(doc), start, balanced_end(doc, start)))
            if first_error is None:
                first_error = error
            continue
        return _decorate(result, payload)

    try:
        return parse_json(text, 0, max_depth=max_depth, complete=True)
    except JSONRepairError:
        if first_error is not None:
            raise first_error from None
        raise JSONRepairError("no JSON value found in the text") from None
