# How markdown-memory-hermes works

## Purpose

This Hermes Agent context-engine plugin replaces old turns in outgoing requests with compact, file-backed working state plus the current turn. Completed turns are preserved in an append-only archive and can be retrieved on demand.

![Hermes context selection, recording, and recall](../assets/flow.svg)

## Request and turn lifecycle

1. Hermes assembles a request with its normal transcript.
2. `select_context` locates the current turn boundary and builds a replacement request containing original system messages, a working-state digest, and messages from the current user turn onward.
3. Tool calls and matching results in the current turn stay together, including parallel and multi-round calls. The prefix is frozen during the turn so retries can reuse stable content.
4. On turn completion, the engine records the completed turn in `transcript.md` and updates `state.md` with capped rolling working state.
5. `context_recall` searches archived turns when older details are needed.

## Files and migration

Each session has `state.md` and `transcript.md` under Hermes' context-memory directory. The plugin does not modify the native session database. `bootstrap.py` reads that database in read-only mode and supports dry-run migration, explicit writes, reconciliation, backfill, and repair.

Migration defaults to dry-run, avoids overwriting existing files, and flags ambiguous session shapes for review. Use repair operations deliberately and retain backups.

## Failure behavior and limits

- With no digest, native context passes through unchanged.
- Hook failures are intended to fail open.
- Request reduction depends on a completed or bootstrapped digest; the first request without prior state is not compressed.
- Older details are recoverable only if archived and found by retrieval.
- Constant-size describes the compact state contribution; a very large current turn can still make a request large.

The description follows `__init__.py`, `bootstrap.py`, tests, and the README. This implementation binds to Hermes-specific interfaces even though the general pattern can be ported.

