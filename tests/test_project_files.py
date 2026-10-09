"""Checks on the release, CI and community files (standard library only)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import structured_guard as sg

ROOT = Path(__file__).resolve().parents[1]
NOREPLY = "335769801+dev-stratum@users.noreply.github.com"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
COMMUNITY = (
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/payload_that_broke.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
)
WORKFLOWS = (".github/workflows/tests.yml", ".github/workflows/release.yml")


def read(name):
    return (ROOT / name).read_text(encoding="utf-8-sig")


def test_version_is_the_same_in_package_metadata_and_changelog():
    pyproject = read("pyproject.toml")
    declared = re.search(r'^version = "([^"]+)"$', pyproject, re.M)
    assert declared and declared.group(1) == sg.__version__
    dated = re.findall(
        r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$",
        read("CHANGELOG.md"),
        re.M,
    )
    assert dated and dated[0][0] == sg.__version__
    versions = [version for version, _ in dated]
    assert len(set(versions)) == len(versions)
    assert "## [Unreleased]" in read("CHANGELOG.md")


def test_release_workflow_publishes_with_trusted_publishing_only():
    text = read(".github/workflows/release.yml")
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert "id-token: write" in text and "environment: pypi" in text
    assert "types: [published]" in text
    assert "GITHUB_REF_NAME" in text and "does not match" in text
    forbidden = ("secrets.", "PYPI_API_TOKEN", "password", "twine upload")
    for word in forbidden:
        assert word not in text, word
    assert text.count("id-token: write") == 1  # only the publishing job


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflows_are_least_privilege_and_safe(name):
    text = read(name)
    assert "permissions:\n  contents: read" in text
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "actions/checkout@v4" in text and "actions/setup-python@v5" in text


def test_ci_covers_three_operating_systems_and_every_supported_python():
    text = read(".github/workflows/tests.yml")
    for runner in ("ubuntu-latest", "windows-latest", "macos-latest"):
        assert runner in text
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f'"{version}"' in text
    assert "experimental: true" in text  # Python 3.14 may not fail the build
    assert "twine check" in text and "coverage report" in text


def test_dependabot_only_watches_github_actions():
    text = read(".github/dependabot.yml")
    assert 'package-ecosystem: "github-actions"' in text
    assert text.count("package-ecosystem") == 1


@pytest.mark.parametrize(
    "name",
    [
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/payload_that_broke.yml",
    ],
)
def test_issue_forms_ask_reporters_to_remove_sensitive_data(name):
    text = read(name)
    assert "API keys" in text and "absolute file paths" in text
    assert "required: true" in text


def test_issue_form_config_points_security_reports_to_private_advisories():
    text = read(".github/ISSUE_TEMPLATE/config.yml")
    url = (
        "https://github.com/dev-stratum/structured-guard"
        "/security/advisories/new"
    )
    assert url in text


def test_security_policy_uses_private_reporting_and_states_the_threat_model():
    text = " ".join(read("SECURITY.md").split())
    assert "private vulnerability reporting" in text
    assert "Do not open a public issue" in text
    assert "never executes model output" in text
    assert "20,000 characters" in text and "max_depth" in text


def test_community_files_contain_no_email_other_than_the_noreply_address():
    for name in COMMUNITY + WORKFLOWS:
        for address in EMAIL.findall(read(name)):
            assert address == NOREPLY, (name, address)


def test_source_distribution_manifest_keeps_tests_and_project_files():
    text = read("MANIFEST.in")
    entries = ("graft tests", "graft .github", "CHANGELOG.md", "SECURITY.md")
    for entry in entries:
        assert entry in text
