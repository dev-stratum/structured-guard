"""Offline unit tests: no network, no API keys, no third-party imports."""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import sys
from pathlib import Path

import pytest

import structured_guard as sg
from structured_guard import parser, sanitizer, validator
from structured_guard.models import dotted, pointer

ROOT = Path(__file__).resolve().parents[1]


def codes(items):
    return [item.code for item in items]


def loads(text, **options):
    return parser.repair_json_text(text, **options).value


def nest(depth, leaf='"leaf"'):
    return '{"n":' * depth + leaf + "}" * depth


def depth_of(value):
    depth = 0
    while isinstance(value, dict):
        value = value["n"]
        depth += 1
    return depth


# ----------------------------------------------------------- parser: basics
def test_valid_json_takes_the_fast_path():
    out = parser.repair_json_text('  {"a": [1, 2.5, "x", null, true]}  ')
    assert out.value == {"a": [1, 2.5, "x", None, True]}
    assert out.repairs == () and out.truncated is False


@pytest.mark.parametrize(
    "text, expected",
    [("42", 42), ('"s"', "s"), ("true", True), ("null", None)],
)
def test_valid_top_level_scalars(text, expected):
    assert loads(text) == expected


def test_argument_validation():
    with pytest.raises(TypeError):
        parser.repair_json_text(b"{}")
    with pytest.raises(ValueError):
        parser.repair_json_text("{}", prefer="objekt")
    with pytest.raises(ValueError):
        parser.repair_json_text("{}", max_depth=0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1', {"a": 1}),
        ('{"a": [1, 2', {"a": [1, 2]}),
        ('{"a": {"b": {"c": [1, {"d": 2', {"a": {"b": {"c": [1, {"d": 2}]}}}),
        ("[[[", [[[]]]),
        ('{"a": [', {"a": []}),
        ('[{"a": 1}, {"b": 2', [{"a": 1}, {"b": 2}]),
        ('{"a": [1, 2,', {"a": [1, 2]}),
    ],
)
def test_unclosed_brackets_are_auto_closed(text, expected):
    out = parser.repair_json_text(text)
    assert out.value == expected
    assert out.truncated is True
    assert codes(out.repairs).count("truncated") == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"ok": tru', {"ok": True}),
        ('{"ok": t', {"ok": True}),
        ('{"ok": fals', {"ok": False}),
        ('{"ok": nul', {"ok": None}),
        ('{"ok": n', {"ok": None}),
        ("{'ok': Tru", {"ok": True}),
        ('[true, fal', [True, False]),
        ('{"n": 12.', {"n": 12}),
        ('{"n": 1.5e', {"n": 1.5}),
        ('{"n": 1.5e-', {"n": 1.5}),
        ('{"n": 42', {"n": 42}),
        ('{"n": -', {}),
        ('{"n": +.', {}),
    ],
)
def test_truncated_primitives_are_reconstructed(text, expected):
    out = parser.repair_json_text(text)
    assert out.value == expected
    assert out.truncated is True


def test_reconstructed_literals_are_reported():
    out = parser.repair_json_text('{"ok": tru')
    assert "reconstructed_literal" in codes(out.repairs)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "b"',
        '{"a": 1, "b":',
        '{"a": 1, "b',
        '{"a": 1,',
        '{"a": 1, "b" ',
    ],
)
def test_dangling_keys_are_dropped(text):
    assert loads(text) == {"a": 1}


def test_truncated_strings_are_closed_and_located():
    out = parser.repair_json_text('{"items": [{"text": "half a sent')
    assert out.value == {"items": [{"text": "half a sent"}]}
    note = [r for r in out.repairs if r.code == "truncated"][0]
    assert note.path == ("items", 0, "text") and "string" in note.message


@pytest.mark.parametrize("tail", ["x\\", "x\\u00", "x\\u", "x\\u0"])
def test_cut_off_escapes_are_dropped(tail):
    assert loads('{"a": "' + tail) == {"a": "x"}


def test_single_quoted_strings():
    out = parser.repair_json_text("{'name': 'Ann', 'tags': ['a', 'b']}")
    assert out.value == {"name": "Ann", "tags": ["a", "b"]}
    assert codes(out.repairs).count("quotes") == 5


def test_smart_quotes_and_unquoted_keys():
    text = '{name: \u201cAnn\u201d, first-name: 1, _id: 2}'
    out = parser.repair_json_text(text)
    assert out.value == {"name": "Ann", "first-name": 1, "_id": 2}
    assert codes(out.repairs).count("unquoted_key") == 3


def test_trailing_commas_are_located():
    out = parser.repair_json_text('{"a": [1, 2,], "b": {"c": 1,},}')
    assert out.value == {"a": [1, 2], "b": {"c": 1}}
    assert [r.path for r in out.repairs] == [("a",), ("b",), ()]
    assert codes(out.repairs) == ["trailing_comma"] * 3


def test_missing_and_stray_commas():
    text = '{"a": 1 "b": 2,, "c": [1 2 {"d": 1} {"e": 2}]}'
    out = parser.repair_json_text(text)
    assert out.value == {"a": 1, "b": 2, "c": [1, 2, {"d": 1}, {"e": 2}]}
    assert {"missing_comma", "extra_comma"} <= set(codes(out.repairs))
    assert loads("[,1,]") == [1] and loads("{,}") == {}
    assert loads('["a" "b"]') == ["a", "b"]


def test_comments_are_removed():
    text = '{\n  // note\n  "a": 1, /* inline */ "b": 2 /* never closed'
    out = parser.repair_json_text(text)
    assert out.value == {"a": 1, "b": 2}
    assert codes(out.repairs).count("comment") == 3
    assert loads('{"a": 1} // done') == {"a": 1}
    assert loads('{"a": 1, // last') == {"a": 1}


def test_python_literals_and_non_finite_numbers():
    text = "{'a': True, 'b': False, 'c': None, 'd': NaN, 'e': -Infinity}"
    out = parser.repair_json_text(text)
    expected = {"a": True, "b": False, "c": None, "d": None, "e": None}
    assert out.value == expected
    assert codes(out.repairs).count("literal") == 3
    assert codes(out.repairs).count("non_finite") == 2
    assert loads("[NaN, 1]") == [None, 1]


def test_non_standard_numbers_are_normalised():
    out = parser.repair_json_text("[007, +5, .5, 5., 1e3, -0.5e2]")
    assert out.value == [7, 5, 0.5, 5, 1000.0, -50.0]
    assert codes(out.repairs).count("number") == 4


@pytest.mark.parametrize(
    "text", ["[1e999]", "[1e999,]", "[" + "9" * 4001 + ",]"]
)
def test_unrepresentable_numbers_fail(text):
    with pytest.raises(sg.JSONRepairError):
        loads(text)


def test_string_escapes_that_models_get_wrong():
    text = r"""{'a': 'it\'s', 'b': 'C\d', 'c': '\u00e9\u12', 'd': '\u00zz'}"""
    out = parser.repair_json_text(text)
    assert out.value == {
        "a": "it's",
        "b": "C\\d",
        "c": "\u00e9\\u12",
        "d": "\\u00zz",
    }
    assert "invalid_escape" in codes(out.repairs)


def test_valid_escapes_inside_lenient_input():
    text = r"""{'a': "x\ny\t\"q\" \/ \\ \u0041 \b\f\r",}"""
    assert loads(text) == {"a": 'x\ny\t"q" / \\ A \b\f\r'}


def test_surrogate_pairs_and_lone_surrogates():
    text = r"{'a': '\ud83d\ude00', 'b': '\ud800!'}"
    assert loads(text) == {"a": "\U0001f600", "b": "\ufffd!"}


def test_raw_control_characters_inside_strings():
    out = parser.repair_json_text('{"a": "line1\nline2\ttab"}')
    assert out.value == {"a": "line1\nline2\ttab"}
    assert "control_characters" in codes(out.repairs)


def test_unescaped_inner_quotes():
    out = parser.repair_json_text('{"q": "He said "hi" to me", "n": 2}')
    assert out.value == {"q": 'He said "hi" to me', "n": 2}
    assert "unescaped_quote" in codes(out.repairs)
    assert loads('["say "yes" now", "ok"]') == ['say "yes" now', "ok"]
    assert loads('{"a": "x" /* c */}') == {"a": "x"}


def test_unterminated_string_after_inner_quote_is_an_error():
    with pytest.raises(sg.JSONRepairError):
        loads('{"a": "x "y" z')


@pytest.mark.parametrize(
    "text",
    [
        '{"a" 1}',
        '{"a": 1 ; "b": 2}',
        "[1 ; 2]",
        '{"a": @}',
        '{"a": abc}',
        "{1: 2}",
        "[1, 2}",
        "",
        "   ",
        "plain words",
        "}",
    ],
)
def test_unrecoverable_input_raises(text):
    with pytest.raises(sg.JSONRepairError):
        loads(text)


def test_error_offsets_are_absolute():
    text = 'intro ```json\n{"a" 1}\n```'
    with pytest.raises(sg.JSONRepairError) as info:
        loads(text)
    assert info.value.offset == text.index("1}")
    assert "offset" in str(info.value)
    assert sg.JSONRepairError("boom").offset is None


def test_a_lenient_scalar_must_fill_the_whole_text():
    assert loads("'hello'") == "hello"
    assert loads("+5") == 5
    with pytest.raises(sg.JSONRepairError):
        loads("None of these apply")


# ------------------------------------------------- parser: deep structures
def test_deeply_nested_objects_parse_and_truncate():
    assert depth_of(loads(nest(60))) == 60
    cut = parser.repair_json_text('{"n":' * 60 + '"le')
    assert depth_of(cut.value) == 60 and cut.truncated


def test_nesting_limit_is_enforced_and_configurable():
    deep = '{"n":' * 65 + "1"
    with pytest.raises(sg.JSONRepairError, match="deeper than 64"):
        loads(deep)
    assert depth_of(loads('{"n":' * 70 + "1", max_depth=70)) == 70


def test_parser_is_not_limited_by_the_recursion_limit():
    depth = sys.getrecursionlimit() * 5
    out = parser.repair_json_text("[" * depth, max_depth=depth)
    levels, node = 1, out.value
    while node:
        node = node[0]
        levels += 1
    assert levels == depth and out.truncated


# ------------------------------------------------------ sanitizer behaviour
def test_markdown_fences_are_unwrapped():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nAnything else?'
    out = parser.repair_json_text(text)
    assert out.value == {"a": 1} and codes(out.repairs) == ["code_fence"]
    assert loads('```\n{"a": 1,}\n```') == {"a": 1}
    assert loads("```JSON\n[1, 2,]\n```") == [1, 2]
    assert loads('```json\n{"a": [1, 2') == {"a": [1, 2]}


def test_fences_for_other_languages_are_ignored():
    text = '```python\nx = [1, 2]\n```\nAnswer: {"a": 1,}'
    assert loads(text) == {"a": 1}
    assert loads('```js\n{"x": 1}\n```\n[5]') == [5]


def test_conversational_filler_is_removed():
    text = 'Sure! Here is the JSON you asked for: {"a": 1} Let me know!'
    out = parser.repair_json_text(text)
    assert out.value == {"a": 1}
    assert codes(out.repairs) == ["leading_text", "trailing_text"]


def test_byte_order_mark_is_not_reported_as_filler():
    out = parser.repair_json_text('\ufeff{"a": 1,}')
    assert out.value == {"a": 1} and codes(out.repairs) == ["trailing_comma"]


def test_strip_to_json_removes_fences_and_filler():
    fenced = 'Sure!\n```json\n{"a": [1, {"b": "}"}]}\n```\nDone'
    assert sanitizer.strip_to_json(fenced) == '{"a": [1, {"b": "}"}]}'
    assert sanitizer.strip_to_json("Result: [1, 2] thanks") == "[1, 2]"
    assert sanitizer.strip_to_json('x {"a": 1') == '{"a": 1'
    with pytest.raises(sg.JSONRepairError):
        sanitizer.strip_to_json("nothing to see")


def test_sanitizer_orders_candidates():
    text = 'See [1]. ```python\nk = {"k": 1}\n``` ```json\n{"a": 1}\n```'
    found = sanitizer.extract_payloads(text, prefer="object")
    assert (found[0].fenced, found[0].text[found[0].start]) == (True, "{")
    loose = [p for p in found if not p.fenced]
    assert loose[0].text[loose[0].start] == "{"
    assert text.index('{"k"') not in [p.start for p in loose]


def test_prefer_selects_the_matching_container():
    text = 'See [1] for details. {"a": 1,}'
    assert loads(text) == [1]
    assert loads(text, prefer="object") == {"a": 1}
    assert loads('{"a": 1,} then [2]', prefer="array") == [2]


def test_fragments_of_a_broken_document_are_never_returned():
    for text in (
        '{"a": 1, "c": @, "d": {"e": 2}}',
        "[1, @, [2]]",
        "{@ 'a}' [1]",
        '{@ "a\\"b" [1]',
        '{@ "unterminated [1]',
        "[[1, @ ]",
    ):
        with pytest.raises(sg.JSONRepairError):
            loads(text)
    assert loads('{name} then {"a": "}" ,}') == {"a": "}"}


def test_balanced_end_ignores_brackets_inside_strings():
    assert sanitizer.balanced_end('{"a": "}"} tail', 0) == 9
    assert sanitizer.balanced_end("[1, [2]", 0) == 7
    assert sanitizer.balanced_end('["a\\"]"]', 0) == 7


# ---------------------------------------------------------------- models
def test_path_rendering():
    path = ("items", 2, "a b", "x~y/z")
    assert dotted(path) == '$.items[2]["a b"]["x~y/z"]'
    assert pointer(path) == "/items/2/a b/x~0y~1z"
    assert dotted(()) == "$" and pointer(()) == ""
    issue = sg.ValidationIssue("type", "bad", ("a", 0))
    assert str(issue) == "$.a[0]: bad" and issue.pointer == "/a/0"


# -------------------------------------------------------------- validator
PERSON = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 10,
            "pattern": "^[A-Z]",
        },
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "score": {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
            "multipleOf": 0.25,
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2,
        },
        "kind": {"enum": ["a", "b", 3, None]},
        "flag": {"const": True},
    },
    "required": ["name"],
    "additionalProperties": False,
}


def test_valid_document_has_no_issues():
    doc = {"name": "Ann", "age": 30, "score": 0.5, "tags": ["x"], "kind": 3}
    assert validator.validate({**doc, "flag": True}, PERSON) == ()


def test_validation_reports_every_problem_with_paths():
    doc = {
        "name": "",
        "age": 200,
        "score": 1,
        "tags": [],
        "kind": "z",
        "flag": False,
        "extra": 1,
    }
    found = {(i.code, i.pointer) for i in validator.validate(doc, PERSON)}
    assert found == {
        ("min_length", "/name"),
        ("pattern", "/name"),
        ("maximum", "/age"),
        ("exclusive_maximum", "/score"),
        ("min_items", "/tags"),
        ("enum", "/kind"),
        ("const", "/flag"),
        ("additional_property", "/extra"),
    }


def test_remaining_bound_violations():
    doc = {"name": "Averyveryverylong", "age": -1, "score": 0, "tags": "abc"}
    doc["tags"] = ["a", "b", "c"]
    found = {i.code for i in validator.validate(doc, PERSON)}
    assert {"max_length", "minimum", "exclusive_minimum", "max_items"} <= found
    off_grid = validator.validate({"name": "A", "score": 0.3}, PERSON)
    assert codes(off_grid) == ["multiple_of"]


def test_missing_required_property_points_at_the_property():
    (issue,) = validator.validate({}, PERSON)
    assert issue.code == "required" and issue.pointer == "/name"


def test_validate_never_changes_anything():
    doc = {"name": "Ann", "age": "30", "extra": 1}
    before = repr(doc)
    found = codes(validator.validate(doc, PERSON))
    assert sorted(found) == ["additional_property", "type"]
    assert repr(doc) == before


def test_type_checks_handle_bool_and_integral_floats():
    check = validator.validate
    assert check(True, {"type": "integer"})[0].code == "type"
    assert check(True, {"type": "number"})[0].code == "type"
    assert check(3.0, {"type": "integer"}) == ()
    assert check(3.5, {"type": "integer"})[0].code == "type"
    assert check(1, {"type": "boolean"})[0].code == "type"
    assert check(None, {"type": "null"}) == ()
    assert check("x", {"type": ["integer", "string"]}) == ()
    assert check(None, {"type": "string", "nullable": True}) == ()
    assert check([1], {"type": "object"})[0].message.startswith("expected obj")


def test_non_finite_and_foreign_values():
    check = validator.validate
    assert check(float("nan"), {"type": "number"})[0].code == "type"
    assert check(float("inf"), {})[0].code == "type"
    assert check((1, 2), {"type": "array"})[0].message.endswith("got tuple")
    huge = check(10**400, {"type": "number", "maximum": 5})
    assert huge[0].code == "maximum"


def test_messages_describe_the_offending_value():
    values = (True, False, 5, 1.5, "s", "x" * 60, [1], {"a": 1})
    shown = [
        validator.validate(v, {"type": "null"})[0].message.split("got ")[1]
        for v in values
    ]
    assert shown[:5] == [
        "boolean true",
        "boolean false",
        "integer 5",
        "number 1.5",
        "string 's'",
    ]
    assert shown[5].endswith("...'") and shown[6] == "array of 1 item(s)"
    assert shown[7] == "object with 1 key(s)"


def test_enum_and_const_use_json_equality():
    check = validator.validate
    assert check(True, {"enum": [1]})[0].code == "enum"
    assert check(1.0, {"enum": [1]}) == ()
    deep = {"a": [1, {"b": None}]}
    assert check(deep, {"const": {"a": [1, {"b": None}]}}) == ()
    assert check({"a": 1}, {"const": {"a": 2}})[0].code == "const"
    assert check({"a": 1}, {"const": {"b": 1}})[0].code == "const"
    assert check([1, 2], {"const": [1]})[0].code == "const"
    assert check([1], {"const": [2]})[0].code == "const"
    assert check("a", {"const": 1})[0].code == "const"
    assert check(99, {"enum": list(range(20))})[0].message.count("...") == 1


def test_boolean_schemas_and_combinators():
    check = validator.validate
    assert check(1, True) == () and check(1, False)[0].code == "false_schema"
    both = {"allOf": [{"type": "integer"}, {"minimum": 3}]}
    assert check(5, both) == () and check(1, both)[0].code == "minimum"
    union = {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 10}]}
    assert check("s", union) == () and check(11, union) == ()
    bad = check(5, union)
    assert bad[0].code == "anyOf"
    assert "2 allowed alternatives" in bad[0].message
    assert len(bad) == 2
    one = {"oneOf": [{"const": 1}, {"const": 2}]}
    assert check(1, one) == () and check(3, one)[0].code == "oneOf"


def test_max_issues_caps_the_report():
    schema = {"items": {"const": 1}}
    issues = validator.validate([0] * 20, schema, max_issues=5)
    assert len(issues) == 5


def test_local_references():
    defs = {
        "node": {
            "type": "object",
            "properties": {"kids": {"type": "array", "items": {"$ref": "#"}}},
            "required": ["kids"],
        },
        "a/b": {"type": "integer"},
        "c d": {"type": "string"},
    }
    tree = {"$ref": "#/$defs/node", "$defs": defs}
    assert validator.validate({"kids": [{"kids": []}]}, tree) == ()
    (issue,) = validator.validate({"kids": [{}]}, tree)
    assert issue.pointer == "/kids/0/kids"
    paths = {
        "properties": {
            "n": {"$ref": "#/$defs/a~1b"},
            "s": {"$ref": "#/$defs/c%20d"},
            "z": {"$ref": "#/anyOf/0"},
        },
        "anyOf": [{"type": "integer"}, {"type": "object"}],
        "$defs": defs,
    }
    assert validator.validate({"n": 1, "s": "x", "z": 5}, paths) == ()
    assert validator.validate({"z": "s"}, paths)[0].pointer == "/z"


def test_recursive_root_reference():
    tree = {
        "type": "object",
        "properties": {"next": {"$ref": "#"}},
        "required": ["v"],
    }
    assert validator.validate({"v": 1, "next": {"v": 2}}, tree) == ()
    (issue,) = validator.validate({"v": 1, "next": {}}, tree)
    assert issue.pointer == "/next/v"


@pytest.mark.parametrize(
    "value, schema",
    [
        ({"a": 1}, "string"),
        ({"a": 1}, 5),
        ({"a": 1}, {"type": "strng"}),
        ({"a": 1}, {"type": []}),
        ({"a": 1}, {"type": 3}),
        ({"a": 1}, {"required": "name"}),
        ({"a": 1}, {"properties": []}),
        (1, {"enum": []}),
        (1, {"enum": "a"}),
        (1, {"anyOf": []}),
        (1, {"allOf": {}}),
        (1, {"$ref": "http://example.test/schema"}),
        (1, {"$ref": "#/nope"}),
        (1, {"$ref": 5}),
        (1, {"$ref": "#/a/b", "a": [1]}),
        (["a"], {"items": [{"type": "string"}]}),
        ("text", {"minLength": -1}),
        ([1], {"minItems": True}),
        (5, {"minimum": "5"}),
        (5, {"multipleOf": 0}),
        ("text", {"pattern": "("}),
        ("text", {"pattern": 5}),
        (1, {"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"}),
    ],
)
def test_invalid_schemas_raise_invalid_schema_error(value, schema):
    with pytest.raises(sg.InvalidSchemaError):
        validator.validate(value, schema)


def test_reference_cycles_are_reported_not_followed_forever():
    cyclic_union = {"anyOf": [{"$ref": "#"}, {"type": "integer"}]}
    with pytest.raises(sg.InvalidSchemaError, match="reference cycle"):
        validator.coerce(1, cyclic_union)


def test_data_deeper_than_max_depth_is_reported():
    tree = {"type": "object", "properties": {"n": {"$ref": "#"}}}
    deep = loads(nest(70, "{}"), max_depth=70)
    (issue,) = validator.validate(deep, tree)
    assert issue.code == "depth" and len(issue.path) == 65
    assert validator.validate(deep, tree, max_depth=100) == ()


# --------------------------------------------------------------- coercion
def test_lossless_conversions_are_applied_and_reported():
    doc = {"name": "Ann", "age": " 42 ", "score": "0.5", "tags": "solo"}
    out = validator.coerce(doc, PERSON)
    expected = {"name": "Ann", "age": 42, "score": 0.5, "tags": ["solo"]}
    assert out.value == expected
    assert out.issues == () and codes(out.repairs) == ["coerced_type"] * 3
    assert out.repairs[0].message == "converted string ' 42 ' to integer"


def test_conversion_table():
    def conv(value, kind):
        return validator.coerce(value, {"type": kind})

    assert conv("7", "integer").value == 7
    assert conv("7.0", "integer").value == 7
    assert conv("7.5", "integer").issues and conv(True, "integer").issues
    assert conv("x" * 5000, "integer").issues
    assert conv("1" * 5000, "number").issues
    assert conv("-3", "number").value == -3
    assert conv("1e2", "number").value == 100.0
    assert conv("nope", "number").issues and conv("1e999", "number").issues
    assert conv("false", "boolean").value is False
    assert conv("maybe", "boolean").issues
    assert conv(12, "string").value == "12"
    assert conv(1.5, "string").value == "1.5"
    assert conv(True, "string").value == "true"
    assert conv(False, "string").value == "false"
    assert conv([1], "string").issues and conv(None, "string").issues
    assert conv(" NULL ", "null").value is None and conv("x", "null").issues
    converted = conv(3.0, "integer")
    assert converted.value == 3 and isinstance(converted.value, int)
    assert converted.repairs[0].code == "coerced_type"
    assert conv(3.0, ["integer", "number"]).value == 3.0


def test_float_stays_float_when_number_is_allowed():
    out = validator.coerce(2.0, {"type": ["number", "integer"]})
    assert out.value == 2.0 and isinstance(out.value, float)
    assert out.repairs == ()


def test_double_encoded_json_is_decoded():
    schema = {
        "properties": {
            "a": {"type": "object"},
            "b": {"type": "array", "items": {"type": "integer"}},
        }
    }
    out = validator.coerce({"a": '{"x": 1}', "b": "[1, 2]"}, schema)
    assert out.value == {"a": {"x": 1}, "b": [1, 2]} and not out.issues


@pytest.mark.parametrize(
    "text",
    ["hello", '{"a": 1} and more', '{"a": 1', "[1, 2]", '{"a" 1}'],
)
def test_prose_is_never_mistaken_for_an_embedded_object(text):
    out = validator.coerce(text, {"type": "object"})
    assert out.issues and out.value == text


def test_arrays_wrap_single_values_but_not_null():
    wrap = {"type": "array"}
    assert validator.coerce("x", wrap).value == ["x"]
    assert validator.coerce({"a": 1}, wrap).value == [{"a": 1}]
    assert validator.coerce('{"a": 1}', wrap).value == ['{"a": 1}']
    assert validator.coerce(None, wrap).issues


def test_type_lists_prefer_exact_matches():
    coerce = validator.coerce
    assert coerce("5", {"type": ["string", "integer"]}).value == "5"
    assert coerce("5", {"type": ["integer", "null"]}).value == 5
    assert coerce("null", {"type": ["integer", "null"]}).value is None
    message = coerce("x", {"type": ["integer", "null"]}).issues[0].message
    assert "expected integer or null" in message


def test_enum_case_insensitive_match_only_when_unique():
    enum = {"enum": ["red", "Blue", "BLUE", 5]}
    assert validator.coerce(" RED ", enum).value == "red"
    assert validator.coerce("blue", enum).issues
    assert validator.coerce("green", enum).issues
    assert validator.coerce(7, enum).issues
    assert validator.coerce("Red", enum).repairs[0].code == "coerced_enum"


def test_union_prefers_exact_then_coerced_branch():
    union = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert validator.coerce("5", union).value == "5"
    assert validator.coerce(True, union).value == "true"
    shape = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    out = validator.coerce({"a": "1", "b": 2}, {"anyOf": [shape]})
    assert out.value == {"a": 1}
    assert codes(out.repairs) == ["coerced_type", "dropped_property"]
    shape_a = {"properties": {"a": {"type": "integer"}}}
    mixed = {"anyOf": [{"type": "string"}, shape_a]}
    best = validator.coerce({"a": "x"}, mixed)
    assert best.issues[0].code == "anyOf" and best.issues[1].pointer == "/a"


def test_objects_defaults_nulls_and_extras():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": ["string", "null"]},
            "c": {"type": "integer", "default": 7},
            "d": {"type": "array", "default": []},
            "r": {"type": "string"},
        },
        "required": ["r"],
        "additionalProperties": False,
    }
    out = validator.coerce({"a": None, "b": None, "r": None, "z": 1}, schema)
    assert out.value == {"b": None, "r": None, "c": 7, "d": []}
    assert [(i.code, i.pointer) for i in out.issues] == [("type", "/r")]
    kinds = {"dropped_null", "dropped_property", "default"}
    assert kinds <= set(codes(out.repairs))
    plain = validator.coerce({"r": "x"}, schema, apply_defaults=False)
    assert plain.value == {"r": "x"}
    kept = validator.coerce({"r": "x", "z": 1}, schema, drop_extra=False)
    assert kept.value == {"r": "x", "z": 1, "c": 7, "d": []}
    assert codes(kept.issues) == ["additional_property"]


def test_defaults_are_copied_not_shared():
    schema = {"properties": {"d": {"default": [1]}}}
    first = validator.coerce({}, schema).value
    first["d"].append(2)
    assert validator.coerce({}, schema).value == {"d": [1]}


def test_additional_properties_schema_and_true():
    schema = {
        "additionalProperties": {"type": "integer"},
        "properties": {"k": {"type": "string"}},
    }
    out = validator.coerce({"k": "v", "x": "1", "y": "z"}, schema)
    assert out.value["x"] == 1 and out.issues[0].pointer == "/y"
    open_schema = {"additionalProperties": True}
    assert validator.coerce({"q": [1]}, open_schema).value == {"q": [1]}


def test_input_is_never_mutated_by_coerce():
    doc = {"name": "Ann", "tags": "x", "extra": {"deep": [1]}}
    snapshot = repr(doc)
    validator.coerce(doc, PERSON)
    assert repr(doc) == snapshot


def test_unsupported_keywords_are_reported():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "format": "email"},
            "not": {"type": "string"},
        },
        "additionalProperties": {"uniqueItems": True},
        "anyOf": [{"if": {}, "then": {}}, {"prefixItems": []}],
        "$defs": {"x": {"minProperties": 1}},
        "patternProperties": {"^a": {"contains": {}}},
    }
    assert validator.unsupported_keywords(schema) == (
        "contains",
        "format",
        "if",
        "minProperties",
        "patternProperties",
        "prefixItems",
        "then",
        "uniqueItems",
    )
    assert validator.unsupported_keywords(PERSON) == ()
    assert validator.unsupported_keywords(True) == ()
    shared = {"type": "string"}
    both = {"properties": {"a": shared, "b": shared}}
    assert validator.unsupported_keywords(both) == ()


# ------------------------------------------------ public API: text in/out
ORDER = {
    "type": "object",
    "properties": {
        "item": {"type": "string"},
        "qty": {"type": "integer", "minimum": 1},
    },
    "required": ["item", "qty"],
    "additionalProperties": False,
}


def test_repair_structured_output_end_to_end():
    text = 'Here you go:\n```json\n{"item": "pen", "qty": "3",}\n```'
    data = sg.repair_structured_output(text, ORDER)
    assert data == {"item": "pen", "qty": 3}


def test_repair_structured_output_without_schema_only_repairs():
    assert sg.repair_structured_output("{'a': [1, 2") == {"a": [1, 2]}


def test_schema_violations_raise_with_details():
    with pytest.raises(sg.SchemaValidationError) as info:
        sg.repair_structured_output('{"item": "pen", "qty": 0}', ORDER)
    error = info.value
    assert error.value == {"item": "pen", "qty": 0}
    assert [(e.code, e.pointer) for e in error.errors] == [("minimum", "/qty")]
    assert "$.qty: must be >= 1" in str(error)


def test_unusable_output_raises_json_repair_error():
    with pytest.raises(sg.JSONRepairError):
        sg.repair_structured_output("I cannot help with that.", ORDER)


def test_exception_hierarchy():
    assert issubclass(sg.JSONRepairError, sg.StructuredGuardError)
    assert issubclass(sg.JSONRepairError, ValueError)
    assert issubclass(sg.SchemaValidationError, sg.StructuredGuardError)
    assert issubclass(sg.SchemaValidationError, ValueError)
    assert issubclass(sg.InvalidSchemaError, ValueError)
    assert issubclass(sg.RefusalError, sg.StructuredGuardError)
    assert issubclass(sg.StructuredGuardError, Exception)
    assert not issubclass(sg.RefusalError, ValueError)


def test_inspect_reports_instead_of_raising():
    text = '{"item": 1, "qty": 0, "x": 1}'
    result = sg.inspect_structured_output(text, ORDER)
    assert not result.ok and result.value == {"item": "1", "qty": 0}
    assert "$.qty: must be >= 1" in result.summary()
    failed = sg.inspect_structured_output("no json at all", ORDER)
    assert failed.value is None and failed.errors[0].code == "parse"
    assert sg.inspect_structured_output('{"item": "a", "qty": 1}', ORDER).ok


def test_retry_prompt_lists_problems_and_caps_length():
    result = sg.inspect_structured_output(
        '{"item": "pen", "qty": 0, "x": 1}', ORDER, drop_extra=False
    )
    text = result.retry_prompt()
    assert text.startswith("Your previous reply did not match")
    assert "- $.qty: must be >= 1; got 0" in text
    assert text.endswith("no commentary and no code fences.")
    many = sg.inspect_structured_output(
        "[" + ",".join(["0"] * 15) + "]", {"items": {"const": 1}}
    )
    assert "... and 5 more" in many.retry_prompt()
    assert "(+12 more)" in many.summary()
    clean = sg.inspect_structured_output('{"item": "a", "qty": 1}', ORDER)
    assert clean.retry_prompt() == ""


def test_unwrap_returns_the_value_when_clean():
    result = sg.inspect_structured_output('{"item": "a", "qty": 2}', ORDER)
    assert result.unwrap() == {"item": "a", "qty": 2}
    assert result.summary() == "no problems"


def test_truncated_output_policy():
    text = '{"item": "pen", "qty": 4, "note": "unfinis'
    lenient = sg.inspect_structured_output(text, ORDER)
    assert lenient.truncated and lenient.value == {"item": "pen", "qty": 4}
    assert lenient.ok
    strict = sg.inspect_structured_output(text, ORDER, allow_truncated=False)
    assert strict.errors[0].code == "parse"
    assert "cut off" in strict.errors[0].message
    with pytest.raises(sg.JSONRepairError, match="cut off"):
        sg.repair_structured_output(text, ORDER, allow_truncated=False)


def test_truncation_is_mentioned_in_the_retry_prompt():
    result = sg.inspect_structured_output('{"item": "pen", "qty": 0', ORDER)
    assert result.truncated and "cut off" in result.retry_prompt()


def test_coerce_false_is_pure_validation():
    result = sg.inspect_structured_output(
        '{"item": "pen", "qty": "3"}', ORDER, coerce=False
    )
    assert not result.ok and result.value == {"item": "pen", "qty": "3"}


def test_schema_hint_picks_the_right_container():
    text = 'Ref [1]. {"item": "pen", "qty": 1}'
    assert sg.repair_structured_output(text, ORDER)["item"] == "pen"
    items = {"items": {"type": "integer"}}
    assert sg.repair_structured_output('{"a": 1,} then [7]', items) == [7]
    hint = {"properties": {}}
    assert sg.repair_structured_output('[1]. {"a": 1,}', hint) == {"a": 1}
    assert sg.repair_structured_output("[1]", True) == [1]


def test_invalid_schema_is_a_programmer_error():
    with pytest.raises(sg.InvalidSchemaError):
        sg.repair_structured_output("{}", {"type": "nonsense"})


# ------------------------------------------------ public API: API responses
def chat(message, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": message}]}


def call(name, arguments, call_id="1"):
    function = {"name": name, "arguments": arguments}
    return {"id": call_id, "type": "function", "function": function}


def test_chat_completion_text():
    content = '```json\n{"item": "pen", "qty": "2"}\n```'
    data = sg.repair_structured_output(chat({"content": content}), ORDER)
    assert data == {"item": "pen", "qty": 2}


def test_chat_completion_tool_calls_and_selection():
    calls = [
        call("other", '{"z": 1}', "1"),
        call("order", '{"item": "pen", "qty": "5"}', "2"),
        call("dict_args", {"item": "cup", "qty": 1}, "3"),
        call("none_args", None, "4"),
        "junk",
        {"id": "5", "type": "function"},
    ]
    response = chat({"content": None, "tool_calls": calls}, "tool_calls")
    fit = sg.repair_structured_output
    assert fit(response, ORDER, tool_name="order") == {"item": "pen", "qty": 5}
    assert fit(response, ORDER, tool_name="dict_args")["item"] == "cup"
    assert not sg.inspect_structured_output(
        response, ORDER, tool_name="none_args"
    ).ok
    assert fit(response, {"type": "object"}) == {"z": 1}
    with pytest.raises(sg.JSONRepairError, match="named 'missing'"):
        fit(response, ORDER, tool_name="missing")


def test_text_wins_over_tool_calls_without_a_tool_name():
    parts = [
        {"type": "text", "text": '{"a": '},
        {"type": "text", "text": "1}"},
        "junk",
        {"type": "image"},
    ]
    message = {"content": parts, "tool_calls": [call("f", "{}")]}
    assert sg.repair_structured_output(chat(message)) == {"a": 1}


def test_refusals_raise_refusal_error():
    refused = chat({"content": None, "refusal": "cannot comply"})
    with pytest.raises(sg.RefusalError, match="cannot comply") as info:
        sg.repair_structured_output(refused, ORDER)
    assert info.value.refusal == "cannot comply"
    with pytest.raises(sg.RefusalError):
        sg.inspect_structured_output(refused, ORDER)


def test_chat_completion_edge_cases():
    read = sanitizer.read_response
    assert read(chat({"content": '{"a": 1'}, finish="length")).truncated
    assert not read(chat({"content": "{}", "refusal": ""})).truncated
    with pytest.raises(sg.JSONRepairError, match="neither text nor"):
        read(chat({"content": "  ", "refusal": None}))
    assert read("plain text") == sanitizer.ModelText("plain text", None, False)


def test_responses_api_shape():
    text_part = {"type": "output_text", "text": '{"item": "pen", "qty": 1}'}
    response = {
        "status": "completed",
        "output": [
            "junk",
            {"type": "reasoning"},
            {"type": "message", "content": ["junk", text_part]},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": " "}],
            },
            {
                "type": "function_call",
                "name": "order",
                "arguments": '{"item": "cup", "qty": 2}',
                "call_id": "c9",
            },
            {"type": "message", "content": "not a list"},
        ],
    }
    fit = sg.repair_structured_output
    assert fit(response, ORDER) == {"item": "pen", "qty": 1}
    assert fit(response, ORDER, tool_name="order") == {"item": "cup", "qty": 2}


def test_responses_api_refusal_and_incomplete():
    refusal = {"type": "refusal", "refusal": "no"}
    refused = {"output": [{"type": "message", "content": [refusal]}]}
    with pytest.raises(sg.RefusalError):
        sg.repair_structured_output(refused)
    read = sanitizer.read_response
    limit = {"reason": "max_output_tokens"}
    cut = {"status": "incomplete", "incomplete_details": limit, "output": []}
    other = {"status": "incomplete", "incomplete_details": {"reason": "x"}}
    with pytest.raises(sg.JSONRepairError):
        read(cut)
    assert sanitizer._read_responses(cut)[2] is True
    assert sanitizer._read_responses({**other, "output": []})[2] is False
    unknown = {"status": "incomplete", "output": []}
    assert sanitizer._read_responses(unknown)[2] is False


class _Dumpable:
    def model_dump(self):
        return chat({"content": '{"a": 1}'})


class _ToDict:
    def to_dict(self):
        return chat({"content": '{"a": 2}'})


class _BadDump:
    def model_dump(self):
        return 5


@pytest.mark.parametrize(
    "obj, expected", [(_Dumpable(), {"a": 1}), (_ToDict(), {"a": 2})]
)
def test_sdk_style_objects_are_supported(obj, expected):
    assert sg.repair_structured_output(obj) == expected


@pytest.mark.parametrize(
    "response",
    [
        None,
        5,
        _BadDump(),
        {},
        {"choices": []},
        {"choices": "x"},
        {"choices": ["x"]},
        {"choices": [{}]},
        {"choices": [{"message": "x"}]},
        {"output": "x"},
    ],
)
def test_unrecognised_responses_raise(response):
    with pytest.raises(sg.JSONRepairError):
        sg.repair_structured_output(response)


# --------------------------------------------------------- guard_json
def test_guard_json_decorator_with_arguments():
    @sg.guard_json(schema=ORDER)
    def ask(prompt):
        return '{"item": "%s", "qty": "2",}' % prompt

    assert ask("pen") == {"item": "pen", "qty": 2}
    assert ask.__name__ == "ask"


def test_guard_json_bare_decorator_and_options():
    @sg.guard_json
    def ask():
        return "{'a': 1}"

    assert ask() == {"a": 1}

    @sg.guard_json(schema=ORDER, tool_name="order")
    def api():
        return chat({"tool_calls": [call("order", '{"item": "x", "qty": 1}')]})

    assert api() == {"item": "x", "qty": 1}


def test_guard_json_propagates_errors():
    @sg.guard_json(schema=ORDER)
    def ask():
        return '{"item": "pen"}'

    with pytest.raises(sg.SchemaValidationError):
        ask()


def test_guard_json_supports_async_functions():
    @sg.guard_json(schema=ORDER)
    async def ask(item):
        return '{"item": "%s", "qty": "4"' % item

    assert asyncio.run(ask("pen")) == {"item": "pen", "qty": 4}
    assert inspect.iscoroutinefunction(ask)


# ------------------------------------------------------ repository hygiene
def _text_files():
    """Project text files, ignoring caches, virtualenvs and local clutter."""
    names = {"LICENSE", ".gitignore"}
    suffixes = {".py", ".md", ".toml", ".yml", ".yaml"}
    found = [
        path
        for path in ROOT.iterdir()
        if path.is_file() and (path.suffix in suffixes or path.name in names)
    ]
    for folder in ("src", "tests", ".github"):
        found.extend(
            path
            for path in (ROOT / folder).rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and "__pycache__" not in path.parts
        )
    return sorted(found)


def _python_files():
    return [p for p in _text_files() if p.suffix == ".py"]


def test_only_standard_library_modules_are_imported():
    allowed = set(sys.stdlib_module_names) | {"pytest", "structured_guard"}
    found = set()
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.add((node.module or "").split(".")[0])
    assert found and found <= allowed, sorted(found - allowed)
    assert "pydantic" not in found and "jsonschema" not in found


def test_python_files_follow_basic_pep8_layout():
    assert len(_python_files()) >= 8
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n"), path.name
        for number, line in enumerate(text.splitlines(), 1):
            where = f"{path.name}:{number}"
            assert len(line) <= 79, where
            assert line == line.rstrip(), where
            assert "\t" not in line, where


def test_repository_has_no_paths_usernames_or_placeholders():
    drive = "C:" + chr(92)
    banned = [drive, "/ho" + "me/", "/Us" + "ers/", "/mn" + "t/", "TO" + "DO"]
    banned += ["FIX" + "ME", "..." + " rest of"]
    bracketed = re.compile(r"<[A-Za-z][A-Za-z_-]*>")
    for path in _text_files():
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token!r} in {path.name}"
        if path.suffix != ".py":
            assert not bracketed.search(text), path.name


def test_package_metadata_matches_the_project_spec():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    email = "335769801+dev-stratum@users.noreply.github.com"
    repo = "https://github.com/dev-stratum/structured-guard"
    assert 'name = "structured-guard"' in pyproject
    assert f'version = "{sg.__version__}"' in pyproject
    assert re.search(r"^dependencies = \[\]$", pyproject, re.M)
    assert 'dev = ["pytest>=7.0.0"]' in pyproject
    assert f'email = "{email}"' in pyproject and repo in pyproject
    assert 'pythonpath = ["src"]' in pyproject
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8-sig")
    assert "Copyright (c) 2026 dev-stratum" in licence
    assert licence.startswith("MIT License")
