# markdown-memory-hermes

A context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that makes long-running agent conversations cost-efficient: instead of re-sending the entire accumulated chat history on every API call, it swaps old history for a compact, file-backed working state.

Works with any model Hermes can drive (OpenAI, Anthropic, Gemini, GLM, DeepSeek, local models — the engine only reshapes the request payload; it never touches model configuration).

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

## Where it sits in a token-reduction stack

markdown-memory is one of three complementary token-reduction layers. They cut cost at different points and stack cleanly:

| Layer | Tool | What it cuts | When it runs |
|---|---|---|---|
| Conversation history | **markdown-memory** | old *turns* — what was said and done before | before every API request |
| Tool output | rtk | the *size of tool results* (shell output, logs), compressed before they enter the transcript | at tool execution time |
| Codebase knowledge | graphify | *file reading* — answers codebase questions from a prebuilt knowledge-graph subgraph instead of grepping/reading source files | during research and exploration |

**Why they stack:** rtk and graphify shrink what *enters* the conversation; markdown-memory shrinks what *stays* in it. Without rtk/graphify, markdown-memory still wins — your digest and current turn are just fatter because tool results arrive uncompressed. Without markdown-memory, rtk/graphify savings get re-billed forever, because everything they saved still rides in the history that every later request re-sends. Together: minimum enters, and nothing accumulates.

## How a request flows through the engine

1. **You send a message** → Hermes assembles the API request: system prompt + full transcript replay + your new message.
2. **Before the request leaves**, the engine's `select_context` hook swaps it for: `system prompt + state.md digest + everything since the last user message`. The digest is frozen at turn boundaries, so every retry and tool-round inside a turn replays a byte-identical prompt prefix (prompt-cache friendly).
3. **When the turn finishes**, `on_turn_complete` deterministically extracts it (user ask, tool calls with ids and results, final response) and appends to `transcript.md`, then rolls `state.md` (last 8 turns, 4k character cap).
4. **If the model needs older detail**, it calls `context_recall`, which greps the archive and returns matching turn sections — you pay only for what you retrieve.
5. **Across sessions and restarts**, `on_session_start` reloads `state.md` — memory survives process death. The session database is never modified.

## Tool-call continuity guarantees

- The cut point is always "before the current turn's user message", so every `tool_call_id` ↔ tool-result pair — including **parallel tool calls** and **multi-round turns** — stays inside the request. Provider pairing rules are never violated (asserted by tests on real wire payloads).
- The digest is frozen mid-turn: byte-identical prefix across all iterations and retries of a turn.
- If the model needs pre-turn history, `context_recall` fetches it on demand instead of the request carrying it.

## Fail-open by design

- Every hook swallows its own exceptions: a broken session directory degrades recording to a no-op while selection keeps working.
- A session without memory files (or with an empty digest) **never** gets its history replaced — the engine passes the request through unchanged, so Hermes' native context behaviour applies.
- The host also wraps `select_context` in its own fail-open guard; a crashing engine cannot break a run.

## Install

```bash
# 1. Copy the plugin into your Hermes plugins directory
cp -r markdown-memory ~/.hermes/plugins/

# 2. Select it in config.yaml
context:
  engine: markdown-memory

# 3. Restart Hermes
```

Rollback: `hermes config set context.engine compressor`, restart Hermes. Memory files stay on disk as inert archives; the session database is untouched at every point.

## Migrating existing sessions

`bootstrap.py` reads the native session database (strictly read-only, `mode=ro`) and creates memory files for existing sessions:

```bash
# Report only — shows what would be written, per session (default: dry-run)
python3 bootstrap.py

# Create memory files for the sessions listed in the report
python3 bootstrap.py --write --session <session-id> ...

# Complete turns that finished after a snapshot (idempotent, append-only)
python3 bootstrap.py --reconcile --write --session <session-id>

# Append whole closed turns that completed after a snapshot (idempotent)
python3 bootstrap.py --reconcile --backfill --write --session <session-id>

# Rebuild state.md from completed history (backs up the old file first)
python3 bootstrap.py --repair-state --write --session <session-id>
```

Safety properties: dry-run default, atomic file creation (tmp + rename), never overwrites existing memory files, live in-flight turns are never touched (closedness = aged, non-empty final assistant row), and ambiguity (orphaned tool results, missing user messages, first row not user) is flagged for manual review instead of silently migrated.

## Porting to other agent frameworks

The *pattern* — pre-send context replacement backed by markdown files — is host-agnostic and model-agnostic. This implementation is Hermes-specific because it binds to three Hermes facilities:

1. the `ContextEngine.select_context()` hook (the seam for replacing a request before it is sent),
2. the session database (for migrating existing history),
3. plugin discovery under `~/.hermes/plugins/`.

Porting to another framework means re-implementing the same contract against that host's interception point: replace accumulated messages with `digest + current turn`, record completed turns on a turn-finalization seam, and expose a retrieval tool. Frameworks without a pre-send seam (most IDE agents) cannot host this pattern without core changes.

## Tests

```bash
HERMES_REPO=/path/to/hermes-agent pytest tests/ -q
```

Includes unit tests (selection semantics, parallel/multi-round tool pairing, determinism, retries, fail-open, migration, reconciliation idempotency) and live-pipeline integration tests that run the real Hermes agent loop against a mock provider. The suite asserts byte-identical retry payloads and zero orphaned tool results on the wire.

## License

MIT
