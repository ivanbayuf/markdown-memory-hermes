# markdown-memory

A context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that makes long-running agent conversations cost-efficient: instead of re-sending the entire accumulated chat history on every API call, it swaps old history for a compact, file-backed working state.

## • What is it

A drop-in **context engine plugin** (no core modifications). Hermes natively replays the full conversation transcript on every LLM request, so input tokens grow with chat length. This engine replaces everything before the current turn with a short digest, keeps the current turn verbatim, and persists what it removed to two files per session:

- `state.md` — compact rolling working memory (goal + recent turns, capped), injected into every request
- `transcript.md` — append-only archive of every turn, searchable on demand via the built-in `context_recall` tool

## • Problem

Long agent sessions become expensive and slow: a chat that runs for days can grow to hundreds of thousands of tokens, and because the full transcript is re-sent on **every** tool call, input-token cost grows quadratically with session activity. Compaction helps, but it fires late, rewrites history, and still ships a large payload.

## • Goal

Make per-request context size **constant** instead of growing with conversation age — without losing durability. All native session history stays intact in Hermes' session database (the plugin is strictly read-only against it); anything dropped from a request is archived to disk and recoverable with `context_recall`.

## • Impact

Benchmarked on a real multi-turn, multi-round-tool conversation (20 turns × 3 API calls): **~81% fewer input tokens** (306k → 59k estimated), with per-request size no longer growing over time (10.2k → 1.4k tokens on the last request vs native). Request size becomes independent of chat length, and giant context stall-retry loops (which re-bill the full payload per retry) disappear.

## Install

```bash
# 1. Copy the plugin into your Hermes plugins directory
cp -r markdown-memory ~/.hermes/plugins/

# 2. Select it in config.yaml
context:
  engine: markdown-memory

# 3. Restart Hermes
```

## Migrating existing sessions

`bootstrap.py` reads your native session database (strictly read-only) and creates memory files for existing sessions:

```bash
# Report only — shows what would be written, per session (default: dry-run)
python3 bootstrap.py

# Create memory files for the sessions listed in the report
python3 bootstrap.py --write --session <session-id> ...

# Complete turns that finished after a snapshot (idempotent, append-only)
python3 bootstrap.py --reconcile --write --session <session-id>

# Rebuild state.md from completed history (backs up the old file first)
python3 bootstrap.py --repair-state --write --session <session-id>
```

Safety properties: dry-run default, SQLite opened `mode=ro`, atomic file creation (tmp + rename), never overwrites existing memory files, live in-flight turns are never touched, ambiguity (orphaned tool results, missing user messages) is flagged instead of silently migrated.

## Design notes

- **Tool-call continuity:** the cut point is always "before the current turn's user message", so every `tool_call_id` ↔ tool-result pair (including parallel calls and multi-round turns) stays inside the request. Provider pairing rules are never violated.
- **Determinism:** the digest is frozen at turn boundaries, so every API call and retry within a turn replays a byte-identical prompt prefix (prompt-cache friendly within the turn).
- **Fail-open:** every hook swallows its own exceptions; a session without memory falls back to native full-history context (an empty digest never replaces history).
- **No auxiliary LLM calls:** the digest is deterministic string extraction — zero extra tokens for summarization.

## Tests

```bash
HERMES_REPO=/path/to/hermes-agent pytest tests/ -q
```

Includes unit tests (selection semantics, parallel/multi-round tool pairing, determinism, retries, fail-open) and live-pipeline integration tests that run the real Hermes agent loop against a mock provider. The suite asserts byte-identical retry payloads and zero orphaned tool results on the wire.

## License

MIT
