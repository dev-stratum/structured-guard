"""Schema validation and conservative coercion, standard library only.

Supported keywords: ``type`` (including lists and ``nullable``), ``enum``,
``const``, ``properties``, ``required``, ``additionalProperties``, ``items``,
``minItems``, ``maxItems``, ``minLength``, ``maxLength``, ``pattern``,
``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``,
``multipleOf``, ``allOf``, ``anyOf``, ``oneOf`` (treated like ``anyOf``),
local ``$ref`` pointers, and ``default`` (used to fill missing properties).
:func:`unsupported_keywords` reports keywords that are present in a schema
but are not enforced.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any
from urllib.parse import unquote

from .exceptions import InvalidSchemaError, JSONRepairError
from .models import DEFAULT_MAX_DEPTH, Path, Repair, SchemaLike
from .models import ValidationIssue

__all__ = ["CoerceResult", "coerce", "unsupported_keywords", "validate"]

_MAX_HOPS = 64
_MAX_TEXT = 4000
_NOPE: Any = object()
_TYPES = frozenset(
    {"null", "boolean", "integer", "number", "string", "array", "object"}
)
_INT_TEXT = re.compile(r"([+-]?[0-9]+)(?:\.0+)?")
_NUM_TEXT = re.compile(
    r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)

_IGNORED_KEYWORDS = frozenset(
    {
        "prefixItems",
        "patternProperties",
        "propertyNames",
        "dependentRequired",
        "dependentSchemas",
        "dependencies",
        "if",
        "then",
        "else",
        "not",
        "contains",
        "minContains",
        "maxContains",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "additionalItems",
        "format",
    }
)
_SCHEMA_SLOTS = (
    "additionalProperties",
    "items",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
)
_LIST_SLOTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_MAP_SLOTS = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
)


@dataclass(frozen=True)
class CoerceResult:
    """Result of :func:`coerce`."""

    value: Any
    repairs: tuple[Repair, ...]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class _Options:
    coerce: bool
    apply_defaults: bool
    drop_extra: bool
    max_issues: int
    max_depth: int


# -- small pure helpers ---------------------------------------------------
def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _matches_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (
            isinstance(value, float) and value.is_integer()
        )
    if name == "number":
        return _is_number(value)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    return isinstance(value, dict)


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean true" if value else "boolean false"
    if isinstance(value, int):
        return f"integer {value!r}"
    if isinstance(value, float):
        return f"number {value!r}"
    if isinstance(value, str):
        shown = value if len(value) <= 40 else value[:37] + "..."
        return f"string {shown!r}"
    if isinstance(value, list):
        return f"array of {len(value)} item(s)"
    if isinstance(value, dict):
        return f"object with {len(value)} key(s)"
    return type(value).__name__


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return bool(a == b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _json_equal(x, y) for x, y in zip(a, b)
        )
    return type(a) is type(b) and bool(a == b)


def _type_names(schema: Mapping[str, Any]) -> list[str]:
    raw = schema.get("type")
    if raw is None:
        return []
    if isinstance(raw, str):
        names = [raw]
    elif (
        isinstance(raw, list)
        and raw
        and all(isinstance(t, str) for t in raw)
    ):
        names = list(raw)
    else:
        raise InvalidSchemaError(f"invalid 'type': {raw!r}")
    unknown = [name for name in names if name not in _TYPES]
    if unknown:
        listed = ", ".join(map(repr, unknown))
        raise InvalidSchemaError(f"unknown type name(s): {listed}")
    if schema.get("nullable") is True and "null" not in names:
        names.append("null")
    return names


def _branches(schema: Mapping[str, Any], name: str) -> list[Any]:
    if name not in schema:
        return []
    items = schema[name]
    if not isinstance(items, list) or not items:
        raise InvalidSchemaError(f"'{name}' must be a non-empty array")
    return items


def _mapping_kw(schema: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = schema.get(name, {})
    if not isinstance(value, Mapping):
        raise InvalidSchemaError(f"'{name}' must be an object")
    return value


def _required_kw(schema: Mapping[str, Any]) -> list[str]:
    value = schema.get("required", [])
    if not isinstance(value, list) or not all(
        isinstance(v, str) for v in value
    ):
        raise InvalidSchemaError("'required' must be an array of strings")
    return value


def _count_kw(schema: Mapping[str, Any], name: str) -> int | None:
    if name not in schema:
        return None
    value = schema[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidSchemaError(f"'{name}' must be a non-negative integer")
    return value


def _number_kw(schema: Mapping[str, Any], name: str) -> int | float | None:
    if name not in schema:
        return None
    value = schema[name]
    if not _is_number(value):
        raise InvalidSchemaError(f"'{name}' must be a number")
    return value  # type: ignore[no-any-return]


def _render_options(options: list[Any]) -> str:
    text = ", ".join(repr(option) for option in options[:8])
    return text + ", ..." if len(options) > 8 else text


# -- coercion primitives: each returns _NOPE when it cannot convert --------
def _to_integer(value: Any) -> Any:
    if isinstance(value, str) and len(value) <= _MAX_TEXT:
        match = _INT_TEXT.fullmatch(value.strip())
        if match:
            return int(match.group(1))
    return _NOPE


def _to_number(value: Any) -> Any:
    if isinstance(value, str) and len(value) <= _MAX_TEXT:
        text = value.strip()
        if _NUM_TEXT.fullmatch(text):
            is_float = any(c in text for c in ".eE")
            number: int | float = float(text) if is_float else int(text)
            if _is_number(number):
                return number
    return _NOPE


def _to_boolean(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return _NOPE


def _to_string(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if _is_number(value):
        return str(value)
    return _NOPE


def _to_null(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return _NOPE


def _decode_embedded(text: str, name: str) -> Any:
    """Decode JSON that a model double-encoded inside a string value."""
    # Imported here because the parser depends on nothing in this module.
    from .parser import repair_json_text

    stripped = text.strip()
    if not stripped.startswith("[" if name == "array" else "{"):
        return _NOPE
    try:
        outcome = repair_json_text(stripped, prefer=name)
    except JSONRepairError:
        return _NOPE
    trailing = any(r.code == "trailing_text" for r in outcome.repairs)
    if outcome.truncated or trailing:
        return _NOPE
    expected = list if name == "array" else dict
    return outcome.value if isinstance(outcome.value, expected) else _NOPE


def _to_container(value: Any, name: str) -> Any:
    if isinstance(value, str):
        decoded = _decode_embedded(value, name)
        if decoded is not _NOPE:
            return decoded
    if name == "array" and value is not None:
        return [value]
    return _NOPE


_CONVERTERS = {
    "integer": _to_integer,
    "number": _to_number,
    "boolean": _to_boolean,
    "string": _to_string,
    "null": _to_null,
}


def _fuzzy_member(value: Any, options: list[Any]) -> Any:
    if not isinstance(value, str):
        return _NOPE
    key = value.strip().casefold()
    hits = [
        option
        for option in options
        if isinstance(option, str) and option.strip().casefold() == key
    ]
    return hits[0] if len(hits) == 1 else _NOPE


def _closeness(engine: _Engine) -> tuple[int, int]:
    """Rank failed union branches: fewest issues, deepest failure first."""
    deepest = max(len(issue.path) for issue in engine.issues)
    return len(engine.issues), -deepest


class _Engine:
    """One validation or coercion run. It never mutates the input value."""

    def __init__(self, root: Any, options: _Options) -> None:
        self.root = root
        self.options = options
        self.repairs: list[Repair] = []
        self.issues: list[ValidationIssue] = []

    def issue(self, code: str, message: str, path: Path) -> None:
        if len(self.issues) < self.options.max_issues:
            self.issues.append(ValidationIssue(code, message, path))

    def note(self, code: str, message: str, path: Path) -> None:
        self.repairs.append(Repair(code, message, path))

    def child(self, mutate: bool) -> _Engine:
        if mutate:
            return _Engine(self.root, self.options)
        exact = replace(
            self.options, coerce=False, apply_defaults=False, drop_extra=False
        )
        return _Engine(self.root, exact)

    def resolve(self, ref: Any) -> Any:
        if not isinstance(ref, str) or not (
            ref == "#" or ref.startswith("#/")
        ):
            raise InvalidSchemaError(
                f"only local '$ref' pointers are supported, got {ref!r}"
            )
        node: Any = self.root
        if ref == "#":
            return node
        for raw in ref[2:].split("/"):
            key = unquote(raw).replace("~1", "/").replace("~0", "~")
            if isinstance(node, Mapping) and key in node:
                node = node[key]
            elif (
                isinstance(node, list)
                and key.isdigit()
                and int(key) < len(node)
            ):
                node = node[int(key)]
            else:
                raise InvalidSchemaError(f"unresolvable '$ref': {ref!r}")
        return node

    def accepts_null(self, schema: Any) -> bool:
        probe = self.child(mutate=False)
        probe.fit(None, schema, (), 0)
        return not probe.issues

    # -- main entry --------------------------------------------------------
    def fit(self, value: Any, schema: Any, path: Path, hops: int) -> Any:
        """Check *value* against *schema*; return the (coerced) value.

        *hops* counts schema indirections ($ref, allOf, anyOf) at the same
        place in the data; it guards against reference cycles. The depth of
        the data itself is ``len(path)``.
        """
        if hops > _MAX_HOPS:
            raise InvalidSchemaError(
                "the schema has a reference cycle or chains more than "
                f"{_MAX_HOPS} references"
            )
        if len(path) > self.options.max_depth:
            self.issue(
                "depth",
                f"nesting is deeper than {self.options.max_depth} levels",
                path,
            )
            return value
        if schema is True:
            return value
        if schema is False:
            self.issue("false_schema", "no value is allowed here", path)
            return value
        if not isinstance(schema, Mapping):
            kind = type(schema).__name__
            raise InvalidSchemaError(
                f"a schema must be an object or a boolean, not {kind}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            self.issue("type", "NaN and Infinity are not valid JSON", path)
            return value
        if "$ref" in schema:
            target = self.resolve(schema["$ref"])
            value = self.fit(value, target, path, hops + 1)
            schema = {k: v for k, v in schema.items() if k != "$ref"}
        for sub in _branches(schema, "allOf"):
            value = self.fit(value, sub, path, hops + 1)
        for keyword in ("anyOf", "oneOf"):
            branches = _branches(schema, keyword)
            if branches:
                value = self.union(value, branches, path, hops, keyword)
        value, ok = self.check_type(value, schema, path)
        if not ok:
            return value
        value = self.check_choices(value, schema, path)
        if isinstance(value, dict):
            return self.check_object(value, schema, path)
        if isinstance(value, list):
            return self.check_array(value, schema, path)
        if isinstance(value, str):
            self.check_string(value, schema, path)
        elif _is_number(value):
            self.check_number(value, schema, path)
        return value

    def union(
        self,
        value: Any,
        branches: list[Any],
        path: Path,
        hops: int,
        keyword: str,
    ) -> Any:
        """anyOf/oneOf: prefer an exact match, then a coerced one."""
        passes = (False, True) if self.options.coerce else (False,)
        failures: list[_Engine] = []
        for mutate in passes:
            for branch in branches:
                trial = self.child(mutate=mutate)
                fitted = trial.fit(value, branch, path, hops + 1)
                if not trial.issues:
                    self.repairs.extend(trial.repairs)
                    return fitted
                failures.append(trial)
        best = min(failures, key=_closeness)
        self.issue(
            keyword,
            f"does not match any of the {len(branches)} allowed alternatives",
            path,
        )
        for found in best.issues:
            self.issue(found.code, found.message, found.path)
        return value

    def check_type(
        self, value: Any, schema: Mapping[str, Any], path: Path
    ) -> tuple[Any, bool]:
        names = _type_names(schema)
        if not names:
            return value, True
        if any(_matches_type(value, name) for name in names):
            if (
                self.options.coerce
                and isinstance(value, float)
                and "number" not in names
            ):
                self.note(
                    "coerced_type",
                    f"converted {_describe(value)} to integer",
                    path,
                )
                return int(value), True
            return value, True
        if self.options.coerce:
            for name in names:
                converter = _CONVERTERS.get(name)
                if converter is None:
                    converted = _to_container(value, name)
                else:
                    converted = converter(value)
                if converted is not _NOPE:
                    self.note(
                        "coerced_type",
                        f"converted {_describe(value)} to {name}",
                        path,
                    )
                    return converted, True
        expected = " or ".join(names)
        found = _describe(value)
        self.issue("type", f"expected {expected}, got {found}", path)
        return value, False

    def check_choices(
        self, value: Any, schema: Mapping[str, Any], path: Path
    ) -> Any:
        if "enum" in schema:
            options = schema["enum"]
            if not isinstance(options, list) or not options:
                raise InvalidSchemaError("'enum' must be a non-empty array")
            if not any(_json_equal(value, option) for option in options):
                fixed = _NOPE
                if self.options.coerce:
                    fixed = _fuzzy_member(value, options)
                if fixed is _NOPE:
                    self.issue(
                        "enum",
                        f"must be one of {_render_options(options)}; "
                        f"got {_describe(value)}",
                        path,
                    )
                else:
                    self.note(
                        "coerced_enum",
                        f"matched {_describe(value)} to the enum value "
                        f"{fixed!r}",
                        path,
                    )
                    value = fixed
        if "const" in schema and not _json_equal(value, schema["const"]):
            self.issue(
                "const",
                f"must equal {schema['const']!r}; got {_describe(value)}",
                path,
            )
        return value

    def check_object(
        self, value: dict[str, Any], schema: Mapping[str, Any], path: Path
    ) -> Any:
        properties = _mapping_kw(schema, "properties")
        required = _required_kw(schema)
        extra = schema.get("additionalProperties", True)
        out: dict[str, Any] = {}
        for key, item in value.items():
            sub_path = path + (key,)
            if key in properties:
                sub = properties[key]
                if (
                    item is None
                    and self.options.coerce
                    and key not in required
                    and not self.accepts_null(sub)
                ):
                    self.note(
                        "dropped_null",
                        f"dropped null for the optional property {key!r}",
                        sub_path,
                    )
                    continue
                out[key] = self.fit(item, sub, sub_path, 0)
            elif extra is False:
                if self.options.drop_extra:
                    self.note(
                        "dropped_property",
                        f"dropped the unexpected property {key!r}",
                        sub_path,
                    )
                else:
                    self.issue(
                        "additional_property",
                        f"unexpected property {key!r}",
                        sub_path,
                    )
                    out[key] = item
            elif extra is True:
                out[key] = item
            else:
                out[key] = self.fit(item, extra, sub_path, 0)
        if self.options.apply_defaults:
            for key, sub in properties.items():
                if (
                    key not in out
                    and isinstance(sub, Mapping)
                    and "default" in sub
                ):
                    out[key] = copy.deepcopy(sub["default"])
                    self.note(
                        "default",
                        f"filled {key!r} with its default value",
                        path + (key,),
                    )
        for key in required:
            if key not in out:
                self.issue(
                    "required",
                    f"missing required property {key!r}",
                    path + (key,),
                )
        return out

    def check_array(
        self, value: list[Any], schema: Mapping[str, Any], path: Path
    ) -> Any:
        items = schema.get("items")
        if isinstance(items, list):
            raise InvalidSchemaError("tuple-form 'items' is not supported")
        out = value
        if items is not None:
            out = [
                self.fit(item, items, path + (index,), 0)
                for index, item in enumerate(value)
            ]
        low = _count_kw(schema, "minItems")
        high = _count_kw(schema, "maxItems")
        if low is not None and len(value) < low:
            self.issue(
                "min_items",
                f"needs at least {low} item(s); got {len(value)}",
                path,
            )
        if high is not None and len(value) > high:
            self.issue(
                "max_items",
                f"allows at most {high} item(s); got {len(value)}",
                path,
            )
        return out

    def check_string(
        self, value: str, schema: Mapping[str, Any], path: Path
    ) -> None:
        low = _count_kw(schema, "minLength")
        high = _count_kw(schema, "maxLength")
        if low is not None and len(value) < low:
            self.issue(
                "min_length",
                f"needs at least {low} character(s); got {len(value)}",
                path,
            )
        if high is not None and len(value) > high:
            self.issue(
                "max_length",
                f"allows at most {high} character(s); got {len(value)}",
                path,
            )
        if "pattern" in schema:
            pattern = schema["pattern"]
            try:
                compiled = re.compile(pattern)
            except (re.error, TypeError) as exc:
                raise InvalidSchemaError(
                    f"invalid 'pattern' {pattern!r}: {exc}"
                ) from None
            if compiled.search(value) is None:
                self.issue(
                    "pattern", f"must match the pattern {pattern!r}", path
                )

    def check_number(
        self, value: int | float, schema: Mapping[str, Any], path: Path
    ) -> None:
        bound = _number_kw(schema, "minimum")
        if bound is not None and value < bound:
            self.issue("minimum", f"must be >= {bound}; got {value!r}", path)
        bound = _number_kw(schema, "maximum")
        if bound is not None and value > bound:
            self.issue("maximum", f"must be <= {bound}; got {value!r}", path)
        bound = _number_kw(schema, "exclusiveMinimum")
        if bound is not None and value <= bound:
            self.issue(
                "exclusive_minimum", f"must be > {bound}; got {value!r}", path
            )
        bound = _number_kw(schema, "exclusiveMaximum")
        if bound is not None and value >= bound:
            self.issue(
                "exclusive_maximum", f"must be < {bound}; got {value!r}", path
            )
        step = _number_kw(schema, "multipleOf")
        if step is not None:
            if step <= 0:
                raise InvalidSchemaError("'multipleOf' must be greater than 0")
            if Decimal(str(value)) % Decimal(str(step)) != 0:
                self.issue(
                    "multiple_of",
                    f"must be a multiple of {step}; got {value!r}",
                    path,
                )


def coerce(
    data: Any,
    schema: SchemaLike,
    *,
    apply_defaults: bool = True,
    drop_extra: bool = True,
    max_issues: int = 50,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> CoerceResult:
    """Coerce *data* towards *schema* and report what could not be fixed.

    The input is never modified. Conversions are conservative and lossless
    only: ``"42"`` to ``42``, ``3.0`` to ``3``, a single value to a
    one-element array, JSON text to an array or object, and case-insensitive
    enum matches. Numeric bounds, lengths and patterns are never "fixed".

    Raises:
        InvalidSchemaError: If *schema* itself is invalid or unsupported.
    """
    options = _Options(True, apply_defaults, drop_extra, max_issues, max_depth)
    engine = _Engine(schema, options)
    fitted = engine.fit(data, schema, (), 0)
    return CoerceResult(fitted, tuple(engine.repairs), tuple(engine.issues))


def validate(
    data: Any,
    schema: SchemaLike,
    *,
    max_issues: int = 50,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[ValidationIssue, ...]:
    """Validate *data* against *schema* without changing anything."""
    options = _Options(False, False, False, max_issues, max_depth)
    engine = _Engine(schema, options)
    engine.fit(data, schema, (), 0)
    return tuple(engine.issues)


def unsupported_keywords(schema: Any) -> tuple[str, ...]:
    """Return the sorted keywords in *schema* that are present but ignored."""
    found: set[str] = set()
    seen: set[int] = set()
    stack: list[Any] = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, Mapping) or id(node) in seen:
            continue
        seen.add(id(node))
        found.update(key for key in node if key in _IGNORED_KEYWORDS)
        stack.extend(node.get(key) for key in _SCHEMA_SLOTS)
        for key in _LIST_SLOTS:
            if isinstance(node.get(key), list):
                stack.extend(node[key])
        for key in _MAP_SLOTS:
            if isinstance(node.get(key), Mapping):
                stack.extend(node[key].values())
    return tuple(sorted(found))
