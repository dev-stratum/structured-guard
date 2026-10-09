# Contributing to structured-guard

Thank you for helping. One person maintains this project, so small, focused
changes are reviewed fastest.

## Ground rules

- Standard library only. There are no runtime dependencies, and pytest is the
  only test dependency.
- Do not import provider SDKs (no `openai`, `anthropic`, `google` or `mcp`
  packages). Payloads are plain dictionaries and strings.
- Tests run offline: no network, no API keys and no live model calls.
- Keep the code compatible with Python 3.10 and keep lines at 79 columns or
  fewer, with no tabs and no trailing whitespace. The test suite checks this.

## Run the tests

```bash
python -m pytest
```

## Report a payload that broke

Use the "Payload that broke" issue form. A minimal payload is best. Before you
paste anything, remove API keys, tokens, personal data, internal host names and
absolute file paths. Shorten long values.

## Pull requests

1. Open an issue first for anything larger than a small fix.
2. Add or update tests. A bug fix starts as a failing test.
3. Add a line under "Unreleased" in `CHANGELOG.md`.
4. Keep each pull request to one change.

## Privacy

This project is maintained under a pseudonymous account. Please do not put
personal information about yourself or others in issues, commits or pull
requests, and do not include machine-specific paths in code, tests or docs.

## Conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
