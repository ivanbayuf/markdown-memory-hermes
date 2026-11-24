# Benchmark evidence and reproduction

## Existing README numbers

The README reports one scenario: 20 turns × 3 API calls, with estimated input tokens of 306,000 for native transcript replay and 59,000 with markdown-memory. Arithmetic from those reported estimates gives an 80.7% reduction (`1 - 59,000 / 306,000`). It also reports a final-request comparison of about 10,200 versus 1,400 estimated tokens.

These are **existing README claims**, not independently rerun measurements in this update. The checked-in tree has tests but no benchmark harness, raw request payloads, tokenizer report, or machine-readable results reproducing this scenario. Treat the figures as estimates, not provider-billed token counts or universal results.

## What tests establish

The checked-in unit and integration tests exercise selection, tool-call pairing, retry-prefix determinism, archival, recall, bootstrap safety, and fail-open behavior. They do not establish the token-reduction figures.

The documented integration command is:

```sh
HERMES_REPO=/path/to/hermes-agent pytest tests/ -q
```

It requires the matching Hermes Agent checkout and test dependencies. This update did not run it because the repository source was available through the GitHub connection only, not as a local checkout.

## Reproduce a comparable token benchmark

1. Pin the Hermes Agent and plugin commits, provider, model, tokenizer, and sampling settings.
2. Use the same 20-turn transcript and three request rounds per turn, including identical tool-call/result payloads in both conditions.
3. Capture outbound messages before provider serialization for native and plugin paths.
4. Count with the pinned tokenizer; report provider input and cached-input usage separately.
5. Repeat with fresh sessions and publish per-request values, total, median, range, and cache conditions.
6. Publish the fixture, runner, environment, raw counts, and checksum.

Do not infer provider cost from character count alone.

