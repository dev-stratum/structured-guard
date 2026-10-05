"""structured-guard: a zero-dependency reliability shield for LLM output.

Repairs, coerces and validates malformed or truncated JSON from structured
outputs, tool calling and the Model Context Protocol, for OpenAI, Claude,
Gemini, MCP and local models, using only the standard library.

>>> from structured_guard import repair_structured_output
>>> repair_structured_output('{"n": 3,} and some chatter')
{'n': 3}
"""

from __future__ import annotations

from .adapters import (
    GuardedCall,
    guard_claude,
    guard_gemini,
    guard_mcp,
    guard_openai,
)
from .core import (
    GuardResult,
    guard_json,
    inspect_structured_output,
    repair_structured_output,
)
from .exceptions import (
    InvalidSchemaError,
    JSONRepairError,
    RefusalError,
    SchemaValidationError,
    StructuredGuardError,
    ToolCallNotFoundError,
    ToolResultError,
)
from .models import Repair, ValidationIssue

__version__ = "0.2.0"

__all__ = [
    "guard_openai",
    "guard_claude",
    "guard_mcp",
    "guard_gemini",
    "guard_json",
    "repair_structured_output",
    "StructuredGuardError",
    "JSONRepairError",
    "SchemaValidationError",
    "RefusalError",
    "InvalidSchemaError",
    "ToolCallNotFoundError",
    "ToolResultError",
    "inspect_structured_output",
    "GuardResult",
    "GuardedCall",
    "ValidationIssue",
    "Repair",
    "__version__",
]
