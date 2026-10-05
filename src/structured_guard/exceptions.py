"""Exception hierarchy for structured_guard.

Every error raised on purpose derives from :class:`StructuredGuardError`, so
one ``except`` clause can catch the whole family.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import ValidationIssue, summarize


class StructuredGuardError(Exception):
    """Base class for every error raised deliberately by this package."""


class JSONRepairError(StructuredGuardError, ValueError):
    """The model output could not be turned into JSON."""

    def __init__(self, message: str, *, offset: int | None = None) -> None:
        text = message if offset is None else f"{message} (at offset {offset})"
        super().__init__(text)
        self.message = message
        self.offset = offset


class SchemaValidationError(StructuredGuardError, ValueError):
    """The repaired data does not satisfy the schema.

    ``errors`` lists every violation found and ``value`` holds the repaired
    (but invalid) data, so callers can still inspect or log it.
    """

    def __init__(
        self, errors: Iterable[ValidationIssue], value: Any = None
    ) -> None:
        self.errors = tuple(errors)
        self.value = value
        super().__init__(summarize(self.errors))


class RefusalError(StructuredGuardError):
    """The model refused to answer instead of producing structured output."""

    def __init__(self, refusal: str) -> None:
        super().__init__(f"the model refused the request: {refusal}")
        self.refusal = refusal


class InvalidSchemaError(StructuredGuardError, ValueError):
    """The schema supplied by the caller is malformed or unsupported."""


class ToolCallNotFoundError(StructuredGuardError, LookupError):
    """The payload holds no tool call that matches the request."""


class ToolResultError(StructuredGuardError):
    """A tool result reports that the tool itself failed."""

    def __init__(self, message: str) -> None:
        super().__init__(f"the tool reported an error: {message}")
        self.message = message
