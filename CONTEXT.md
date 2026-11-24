# Context: Why markdown-memory exists

## Problem

The orchestrator model (glm-5.3-flash) burned far more tokens than the worker agents (Claude Code, Codex, DeepSeek) — the opposite of what you'd expect, since the orchestrator mostly routes while workers do the heavy lifting.

Root cause: Hermes re-sent the **entire accumulated conversation history** to the LLM on every request. In a long session with many tool calls, each request carried the whole history again, so input tokens grew quadratically. The orchestrator runs 24/7 across many sessions, so it compounded worst.

## Why this solution

We tested whether Hermes could operate statelessly between tool calls. Three constraints shaped the design:

- **No core runtime changes** — must be a plugin that intercepts the message-building pipeline and fails open to native behavior if it breaks.
- **No auxiliary LLM summarizer** — an LLM rewriting history would itself cost tokens and add nondeterminism. So: deterministic, code-only context selection.
- **Preserve everything** — native history/database untouched; tool-call continuity must hold exactly across parallel calls and retries.

That pointed to: keep only the **current task context** in the live conversation, store a compact digest of the past in `state.md`, and archive the full transcript to `transcript.md` on disk — anything older is retrieved on demand instead of being carried in every request.

## How

1. **`markdown-memory` plugin** — a `ContextEngine` subclass replacing the native engine. Each turn it injects the `state.md` digest + working memory in place of full history, and appends new turns to the archive. Old sessions are migrated with `bootstrap.py` (dry-run first, then verified backfill).
2. **Token-reduction refinements** — keyword-based archive retrieval instead of scanning whole turns, leaner turn summaries, and a cap on the goal line in the injected context.
