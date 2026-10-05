"""Small value types shared by every structured_guard module."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

DEFAULT_MAX_DEPTH = 64

PathSegment = Union[str, int]
Path = tuple[PathSegment, ...]
SchemaLike = Union[Mapping[str, Any], bool]


def pointer(path: Path) -> str:
    """Render *path* as an RFC 6901 JSON Pointer ('' is the root)."""
    return "".join(
        "/" + str(seg).replace("~", "~0").replace("/", "~1") for seg in path
    )


def dotted(path: Path) -> str:
    """Render *path* compactly, for example ``$.items[2].name``."""
    out = "$"
    for seg in path:
        if isinstance(seg, int):
            out += f"[{seg}]"
        elif seg.isidentifier():
            out += f".{seg}"
        else:
            out += "[" + json.dumps(seg) + "]"
    return out


@dataclass(frozen=True)
class Finding:
    """A message attached to a location inside a JSON document."""

    code: str
    message: str
    path: Path = ()

    @property
    def pointer(self) -> str:
        """The location as an RFC 6901 JSON Pointer."""
        return pointer(self.path)

    def __str__(self) -> str:
        return f"{dotted(self.path)}: {self.message}"


class Repair(Finding):
    """A change made to the model output so that it could be used."""


class ValidationIssue(Finding):
    """A problem that is still present after all repairs."""


def summarize(issues: Sequence[Finding], limit: int = 3) -> str:
    """One-line summary of *issues*, showing at most *limit* of them."""
    if not issues:
        return "no problems"
    shown = "; ".join(str(issue) for issue in issues[:limit])
    more = f" (+{len(issues) - limit} more)" if len(issues) > limit else ""
    return f"{len(issues)} problem(s): {shown}{more}"
