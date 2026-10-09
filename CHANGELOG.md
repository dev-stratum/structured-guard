# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-10-09

### Added
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md` and `CODE_OF_CONDUCT.md`.
- Issue forms for bug reports and for payloads that broke, a pull request
  template, and a Dependabot configuration that keeps GitHub Actions current.
- Continuous integration on Linux, Windows and macOS for Python 3.10 to 3.13,
  an experimental Python 3.14 run, a branch-coverage report, and a job that
  builds the package, checks its metadata and imports the built wheel.
- A release workflow that publishes to PyPI with Trusted Publishing (no
  stored tokens) after checking that the tag matches the package version.
- `MANIFEST.in`, so source distributions include the tests and project files.
- Tests for the release, CI and community files.

### Changed
- The README installs from PyPI first, and its links work on the PyPI page.
- The coverage configuration points at the source directory.

### Security
- Rebuilding a Gemini `MALFORMED_FUNCTION_CALL` now refuses call text longer
  than 20,000 characters or nested deeper than 50 levels, because the standard
  library warns that parsing such text with `ast` can exhaust the stack.

## [0.2.0] - 2026-10-05

### Added
- Provider adapters `guard_openai`, `guard_claude`, `guard_mcp` and
  `guard_gemini`, with `guard_*_calls` variants for parallel tool calls.
- Support for OpenAI Chat Completions and Responses API payloads, Claude
  `tool_use` blocks, MCP `tools/call` requests and `CallToolResult` payloads,
  Gemini `function_call` parts, Ollama chat responses and tool calls that
  local models print as text.
- Typed `ToolCallNotFoundError` and `ToolResultError` exceptions.
- Repair of Gemini `MALFORMED_FUNCTION_CALL` messages, parsed with `ast` and
  never executed.

### Changed
- The package version moved from the internal 0.1 baseline to 0.2.0.
