"""markdown-memory — a file-backed, semi-stateless context engine plugin for Hermes.

Installed as a user context engine (``$HERMES_HOME/plugins/markdown-memory/``) and
selected with::

    context:
      engine: markdown-memory

in ``config.yaml``. Selecting it never changes the host runtime: the persisted
transcript / session DB is untouched, and this engine only changes the per-request
wire payload.

Design
------
* ``state.md`` — compact working memory (deterministically extracted per turn,
  capped), injected in place of all pre-current-turn history on every request.
* ``transcript.md`` — append-only full archive of every turn (never sent to the
  API). Recoverable through the ``context_recall`` tool without loading history.
* ``select_context()`` keeps: leading system message(s) + a frozen digest message
  + everything from the current turn's user message onward. The current turn's
  assistant tool_calls and matching tool results therefore ALWAYS travel together,
  satisfying provider tool-call pairing on every request — including parallel tool
  calls and multiple tool rounds.
* The digest is frozen at turn boundaries (``on_turn_complete``), never mid-turn,
  so the request prefix is byte-stable across all iterations and retries of a turn
  (prompt-cache friendly within the turn).
* Fully fail-open: every hook swallows its own exceptions and degrades to
  no-op / ``None``; the host additionally wraps ``select_context`` in its own
  fail-open guard, so any plugin failure falls back to native context behaviour.

No auxiliary LLM summarization is used: the digest is deterministic string
extraction (user ask, tools used, assistant final response, status).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("plugins.markdown_memory")

ENGINE_NAME = "markdown-memory"

try:
    from agent.context_engine import ContextEngine as _ContextEngineBase
except Exception:  # pragma: no cover - import-environment safety net
    from abc import ABC as _ABC

    class _ContextEngineBase(_ABC):  # type: ignore[no-redef]
        name = "markdown-memory"

        def update_from_response(self, usage: Dict[str, Any]) -> None:
            pass

        def should_compress(self, prompt_tokens=None) -> bool:
            return False

        def compress(self, messages, current_tokens=None, **kwargs):
            return messages


DIGEST_PREFIX = (
    "[markdown-memory working state — earlier conversation was archived to disk "
    "and replaced here. Prior details are recoverable with the context_recall tool.]\n\n"
)

_TURN_SECTION_RE = re.compile(r"^### Turn ", re.MULTILINE)


def _flatten(content: Any) -> str:
    """Visible text of a message content shape (str, list-of-parts, None)."""
    try:
        from agent.message_content import flatten_message_text
        return flatten_message_text(content) or ""
    except Exception:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            return "\n".join(parts)
        return ""


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _tool_names(msg: dict[str, Any]) -> List[str]:
    calls = msg.get("tool_calls") or []
    names = []
    for c in calls:
        fn = c.get("function", {}) if isinstance(c, dict) else {}
        name = fn.get("name", "?")
        names.append(str(name))
    return names


class MarkdownMemoryEngine(_ContextEngineBase):
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        max_state_chars: int = 4000,
        max_recent_turns: int = 8,
        turn_user_chars: int = 300,
        turn_assistant_chars: int = 500,
        archive_msg_chars: int = 2000,
    ):
        self.max_state_chars = max_state_chars
        self.max_recent_turns = max_recent_turns
        self.turn_user_chars = turn_user_chars
        self.turn_assistant_chars = turn_assistant_chars
        self.archive_msg_chars = archive_msg_chars
        self._session_dir: Optional[Path] = None
        self._session_id: Optional[str] = None
        # Frozen digest: only mutated at turn boundaries / session start, NEVER
        # mid-turn, so every request within a turn shares a byte-identical prefix.
        self._digest: str = ""
        self._turn_counter: int = 0

    @property
    def name(self) -> str:
        return ENGINE_NAME

    # Token state read by the host.
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0
    threshold_percent: float = 1.0
    protect_first_n: int = 3
    protect_last_n: int = 0
    emit_automatic_compaction_status: bool = False

    # ------------------------------------------------------------- lifecycle
    def on_session_start(self, session_id: str, hermes_home: str = "", **kwargs) -> None:
        """Load (or create) this session's markdown state; freeze the digest."""
        try:
            base = Path(hermes_home) if hermes_home else Path(
                os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
            )
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))[:120] or "default"
            self._session_dir = base / "context-memory" / safe
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._session_id = session_id
            self._turn_counter = self._count_archived_turns()
            self._digest = self._load_state_text()
            self._ensure_archive()
        except Exception:
            logger.warning("markdown-memory on_session_start failed; engine goes passive", exc_info=True)
            self._session_dir = None
            self._digest = ""

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        return

    def on_session_reset(self) -> None:
        # /new or /reset: the engine keeps its files (history is durable), but the
        # in-memory frozen digest is dropped; a fresh state file begins on demand.
        # The turn counter is reset too: with the digest gone, select_context()
        # must pass requests through unchanged until a new turn is recorded.
        # (Previously only the digest was cleared, so a reset session could have
        # its history replaced by a placeholder digest. Fixed 2026-11-24.)
        # Note: turn numbers in transcript.md restart at 1 after a reset; the
        # archive count is reconciled by _count_archived_turns() on the next
        # on_session_start().
        self._digest = ""
        self._turn_counter = 0

    # ------------------------------------------------------------- token API
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        try:
            self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
            self.last_total_tokens = int(usage.get("total_tokens") or 0)
        except Exception:
            pass

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        # This engine owns compaction policy: it never rewrites the persisted
        # transcript, so the host's built-in compaction must never fire.
        return False

    def compress(
        self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        # Manual /compress is a no-op: request context is already replaced per call.
        return messages

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return False

    # ------------------------------------------------------- context selection
    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        incoming_message: Optional[Dict[str, Any]] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Replace pre-current-turn history with the frozen digest; keep the
        current turn verbatim from its user message onward. Fail-open: any
        problem returns ``None`` (request unchanged)."""
        try:
            return self._select(request_messages)
        except Exception:
            logger.warning("markdown-memory select_context failed; request unchanged", exc_info=True)
            return None

    def _select(self, request_messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if not request_messages:
            return None
        # Safety net: NEVER replace history with an empty digest. A pre-engine
        # session (or one whose recording failed, or one just reset) must fall
        # back to native context. There is deliberately no placeholder digest:
        # an earlier version substituted "(no prior context recorded...)" here,
        # which discarded real history after on_session_reset() cleared the
        # digest while the turn counter stayed > 0 (fixed 2026-11-24).
        if not self._digest.strip():
            return None
        last_user_idx: Optional[int] = None
        for i, msg in enumerate(request_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = i
        if last_user_idx is None or last_user_idx <= 1:
            # First request of the session (system [+ one message]): nothing to trim.
            return None
        # Leading system message(s) are always preserved.
        head: List[Dict[str, Any]] = []
        for msg in request_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                head.append(msg)
            else:
                break
        digest = self._digest.strip()
        digest_msg = {"role": "user", "content": DIGEST_PREFIX + digest}
        return [*head, digest_msg, *request_messages[last_user_idx:]]

    # ------------------------------------------------------- turn recording
    def on_turn_complete(
        self,
        messages: List[Dict[str, Any]],
        usage: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Deterministically extract the finished turn into state.md and append the
        full window to transcript.md. Best-effort: never raises."""
        try:
            self._record_turn(messages, kwargs)
        except Exception:
            logger.warning("markdown-memory on_turn_complete failed; archive may lag", exc_info=True)

    def _current_turn_window(self, messages: List[Dict[str, Any]]) -> tuple[int, List[Dict[str, Any]]]:
        last_user_idx = None
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = i
        if last_user_idx is None:
            return -1, []
        return last_user_idx, messages[last_user_idx:]

    def _record_turn(self, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        if self._session_dir is None:
            # Passive engine (session start failed): record nothing rather than crash.
            return
        start, window = self._current_turn_window(messages)
        if not window:
            return
        self._turn_counter += 1
        turn_no = self._turn_counter
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "ok"
        if kwargs.get("failed"):
            status = "failed"
        if kwargs.get("interrupted"):
            status = "interrupted"

        # --- archive entry (append-only; never rewritten)
        lines = [f"## Turn {turn_no} — {stamp} — status: {status}"]
        for msg in window:
            role = msg.get("role", "?")
            if role == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    fn = call.get("function", {}) if isinstance(call, dict) else {}
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        args_text = args
                    else:
                        args_text = json.dumps(args, ensure_ascii=False, default=str)
                    lines.append(
                        f"TOOL_CALL {fn.get('name', '?')} args: {_clip(args_text, self.archive_msg_chars)}"
                    )
            text = _flatten(msg.get("content"))
            if role == "tool":
                tid = msg.get("tool_call_id", "")
                name = msg.get("name", "")
                label = tid if tid else name
                if tid and name:
                    label = f"{tid} ({name})"
                lines.append(f"TOOL_RESULT[{label}]: {_clip(text, self.archive_msg_chars)}")
            elif role in ("user", "assistant"):
                if text:
                    lines.append(f"{role.upper()}: {_clip(text, self.archive_msg_chars)}")
        entry = "\n".join(lines).rstrip() + "\n"
        self._append_archive(entry)

        # --- working state (rolling, deterministic)
        user_text = next((_flatten(m.get("content")) for m in window if m.get("role") == "user"), "")
        assistant_text = ""
        tool_list: List[str] = []
        for m in window:
            tool_list.extend(_tool_names(m))
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                assistant_text = _flatten(m.get("content")) or assistant_text
        turn_digest = (
            f"### Turn {turn_no} ({stamp})\n"
            f"User: {_clip(user_text, self.turn_user_chars)}\n"
            f"Tools used: {', '.join(tool_list[:10]) or 'none'}\n"
            f"Assistant: {_clip(assistant_text, self.turn_assistant_chars) or '(no final text)'}\n"
        )
        self._digest = self._write_state(turn_digest)

    # ------------------------------------------------------------ file plumbing
    def _state_path(self) -> Path:
        assert self._session_dir is not None
        return self._session_dir / "state.md"

    def _archive_path(self) -> Path:
        assert self._session_dir is not None
        return self._session_dir / "transcript.md"

    def _ensure_archive(self) -> None:
        if not self._archive_path().exists():
            self._archive_path().write_text(
                f"# Transcript archive — session {self._session_id}\n"
                "Append-only. Never sent to the API; search it with context_recall.\n",
                encoding="utf-8",
            )

    def _append_archive(self, entry: str) -> None:
        self._ensure_archive()
        with self._archive_path().open("a", encoding="utf-8") as fh:
            fh.write("\n" + entry)

    def _count_archived_turns(self) -> int:
        try:
            return len(re.findall(r"(?m)^## Turn ", self._archive_path().read_text(encoding="utf-8")))
        except Exception:
            return 0

    def _load_state_text(self) -> str:
        try:
            return self._state_path().read_text(encoding="utf-8")
        except Exception:
            return ""

    def _write_state(self, new_turn_digest: str) -> str:
        """Rolling rewrite of state.md: header + goal + recent turns, capped."""
        existing = self._load_state_text()
        goal = ""
        matches = list(re.finditer(r"^## Goal\n(.*?)(?=^## |\Z)", existing, re.MULTILINE | re.DOTALL))
        if matches:
            goal = matches[0].group(1).strip()
        elif self._digest:
            pass
        if not goal:
            # First recorded turn of the session: derive the goal from it.
            user_line = ""
            m = re.search(r"^User: (.*)$", new_turn_digest, re.MULTILINE)
            if m:
                user_line = m.group(1)
            goal = _clip(user_line, 300)
        # Keep only the most recent N turn sections from the previous digest.
        # NOTE: _TURN_SECTION_RE.split() discards the "### Turn " delimiter, so
        # each kept section must get its heading back -- otherwise older turns
        # lose their headings on every rewrite and the rolling cap stops
        # counting turns (fixed 2026-11-24; see test_write_state_keeps_headings).
        # Re-adding the heading also repairs state files degraded by the old
        # bug, whose sections look like "1 (2026-...)" without a heading.
        old_turns = _TURN_SECTION_RE.split(existing.split("## Recent turns", 1)[-1]) if "## Recent turns" in existing else []
        old_turns = ["### Turn " + t for t in old_turns if t.strip()]
        kept = old_turns[-(self.max_recent_turns - 1):]
        header = (
            "# Working state (markdown-memory)\n"
            f"Session: {self._session_id}\n"
            f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## Goal\n{goal or '(unknown)'}\n\n"
            "## Recent turns\n"
        )
        new_turns = kept + [new_turn_digest]
        text = header + "".join(new_turns)
        # Hard cap: drop oldest turns until it fits.
        while len(text) > self.max_state_chars and len(new_turns) > 1:
            new_turns.pop(0)
            text = header + "".join(new_turns)
        tmp = self._state_path().with_suffix(".md.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self._state_path())
        return text

    # -------------------------------------------------------------- recall tool
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "context_recall",
                "description": (
                    "Search this conversation's archived history (earlier turns were "
                    "archived to disk instead of being carried in context). Use it to "
                    "recover details from before the current task window."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to look for in the archived transcript.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max matching turns to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if name != "context_recall":
                return json.dumps({"error": f"Unknown context engine tool: {name}"})
            return self._recall(args or {})
        except Exception as exc:
            return json.dumps({"error": f"context_recall failed: {exc}"})

    def _recall(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        limit = max(1, min(int(args.get("limit") or 5), 20))
        if self._session_dir is None:
            return json.dumps({"matches": [], "note": "archive unavailable"})
        try:
            raw = self._archive_path().read_text(encoding="utf-8")
        except Exception:
            return json.dumps({"matches": [], "note": "archive not readable"})
        # Split into turn sections on the "## Turn N" headings.
        sections: List[str] = []
        for block in re.split(r"(?m)^(?=## Turn )", raw):
            if block.startswith("## Turn "):
                sections.append(block.strip())
        if not query:
            hits = sections[-limit:]
            return json.dumps({"matches": hits, "total_turns": len(sections)})
        terms = [t.lower() for t in re.split(r"\s+", query) if t]
        scored = []
        for sec in sections:
            low = sec.lower()
            score = sum(low.count(t) for t in terms)
            if score:
                scored.append((score, sec))
        scored.sort(key=lambda pair: (-pair[0], sections.index(pair[1])))
        matches = [sec for _, sec in scored[:limit]]
        return json.dumps({"matches": matches, "total_turns": len(sections)})

    # ---------------------------------------------------------------- status
    def get_status(self) -> Dict[str, Any]:
        last_prompt = max(self.last_prompt_tokens, 0)
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": min(100, last_prompt / self.context_length * 100) if self.context_length else 0,
            "compression_count": self.compression_count,
        }


def register(ctx) -> None:
    """Plugin-system registration (also used when loaded via the general plugin path)."""
    ctx.register_context_engine(MarkdownMemoryEngine())


# The loader also accepts a bare ``ContextEngine`` subclass or instance exported as
# ``ENGINE``; register(ctx) covers the plugin-host path, and a module-level instance
# covers the direct ``load_named`` capture path.
ENGINE = MarkdownMemoryEngine()
