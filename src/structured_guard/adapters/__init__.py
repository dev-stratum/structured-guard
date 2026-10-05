"""Provider adapters: one guard per tool-calling dialect.

Each ``guard_*`` function takes the raw payload of one provider, repairs
it, validates it against a JSON Schema and returns a clean dictionary. Each
``guard_*_calls`` function does the same for every call in the payload.
"""

from __future__ import annotations

from ._common import GuardedCall
from .claude import guard_claude, guard_claude_calls
from .gemini import (
    guard_gemini,
    guard_gemini_calls,
    normalize_gemini_schema,
    recover_gemini_calls,
)
from .mcp import guard_mcp, guard_mcp_calls
from .openai import guard_openai, guard_openai_calls

__all__ = [
    "GuardedCall",
    "guard_claude",
    "guard_claude_calls",
    "guard_gemini",
    "guard_gemini_calls",
    "guard_mcp",
    "guard_mcp_calls",
    "guard_openai",
    "guard_openai_calls",
    "normalize_gemini_schema",
    "recover_gemini_calls",
]
