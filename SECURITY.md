# Security policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately with GitHub's private vulnerability
reporting: open the repository's **Security** tab and choose **Report a
vulnerability**. Do not open a public issue for a security problem. I aim to
acknowledge a report within 7 days.

## What counts as a vulnerability

structured-guard parses untrusted model output, so these are in scope:

- input that makes parsing take super-linear time, such as regular-expression
  or algorithmic-complexity attacks;
- input that exhausts memory or the interpreter's stack;
- any path that executes model-supplied text or touches the file system or the
  network;
- an exception other than a `StructuredGuardError` escaping from a `guard_*`
  function for hostile input.

## Threat model

- The library makes no network calls, reads no files and never executes model
  output. JSON is parsed with a stack-based parser, and the one place that
  reads Python-style call text (rebuilding a Gemini `MALFORMED_FUNCTION_CALL`)
  uses `ast.parse` and `ast.literal_eval` on literal keyword arguments only,
  and refuses text longer than 20,000 characters or nested deeper than 50
  levels.
- Nesting is limited by `max_depth` (64 by default). A limit on total input
  size is planned for a later release; until then, bound the size of what you
  pass in.
- Schemas are trusted input. Do not build schemas from untrusted text.
