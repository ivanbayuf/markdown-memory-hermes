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
from collections import Counter
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
    "[markdown-memory: earlier turns archived to disk; recover via context_recall.]\n\n"
)

# Splitting on a zero-width lookahead keeps the "### Turn N" heading attached
# to each kept turn section (a consuming split stripped headings from old turns).
_TURN_SECTION_RE = re.compile(r"(?m)^(?=### Turn )")

_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been being it its this "
    "that these those i you my your we our not no do does did done don doesn what how "
    "why when where which who whom so if then than there here at as by from but can "
    "could will would should may might just have has had about into out up down over "
    "under more most some any all also very much many now new one two first next last "
    "want need like get got put use used using make made run ran okay yes yeah let lets "
    "them their they he she his her him its itself are was were am been ok im ive dont "
    "cant wont theres whatre that's it's i'm you're we're".split()
)


def _keywords(text: str, limit: int = 8) -> str:
    """Deterministic keyword extraction: frequent non-stopword tokens."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (text or "").lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return ", ".join(w for w, _ in counts.most_common(limit))


def _scan_archive(text: str) -> List[Dict[str, Any]]:
    """Map every archived turn to its transcript.md line range.

    Only real ``## Turn N`` headings delimit turns; ``#### Turn N — completion``
    blocks (bootstrap reconcile) are merged into turn N wherever they appear.
    Returns a list of dicts: {no, start, end, heading, user, assistant, completion}.
    ``start``/``end`` are 1-based transcript.md line numbers (end inclusive).
    """
    lines = text.splitlines()
    turns: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    current: Optional[Dict[str, Any]] = None
    for lineno, raw in enumerate(lines, start=1):
        m = re.match(r"^## Turn (\d+) — (.*?) — status: (\S+)", raw)
        if m:
            current = {"no": int(m.group(1)), "heading": raw, "start": lineno,
                       "end": lineno, "user": "", "assistant": "",
                       "extra_headings": []}
            turns[current["no"]] = current
            order.append(current["no"])
            continue
        m = re.match(r"^#### Turn (\d+) — completion", raw)
        if m and int(m.group(1)) in turns:
            turns[int(m.group(1))]["extra_headings"].append((lineno, raw))
        if current is None:
            continue
        current["end"] = lineno
        if raw.startswith("USER: ") and not current["user"]:
            current["user"] = raw[len("USER: "):]
        elif raw.startswith("ASSISTANT: "):
            current["assistant"] = raw[len("ASSISTANT: "):] or current["assistant"]
    return [turns[no] for no in order]


def build_index(session_dir: Path) -> str:
    """(Re)build index.md from transcript.md; return the index text.

    One entry per archived turn: heading, first user line, last assistant line,
    keywords, and the exact transcript.md line range for pinpoint reading.
    Deterministic — same transcript always yields the same index.
    """
    transcript_path = Path(session_dir) / "transcript.md"
    try:
        text = transcript_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    session_id = ""
    m = re.search(r"^# Transcript archive — session (.*)$", text, re.MULTILINE)
    if m:
        session_id = m.group(1).strip()
    entries = _scan_archive(text)
    out = [
        "# Index — session " + (session_id or "unknown"),
        "Compact map of transcript.md. Find a topic here, then read the cited",
        "transcript.md lines (or context_recall turn=N for the full block).",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')} — turns: {len(entries)}",
        "",
    ]
    for e in entries:
        block_text = "\n".join(text.splitlines()[e["start"] - 1:e["end"]])
        out.append(
            f"### Turn {e['no']} — transcript.md lines {e['start']}-{e['end']}"
            + (f" (+{len(e['extra_headings'])} completion block(s))" if e["extra_headings"] else "")
        )
        heading_m = re.match(r"^## Turn \d+ — (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) — status: (\S+)", e["heading"])
        when, status = (heading_m.group(1), heading_m.group(2)) if heading_m else ("?", "?")
        out.append(f"when: {when} | status: {status}")
        out.append(f"User: {_clip(e['user'], 200)}")
        out.append(f"Assistant: {_clip(e['assistant'], 200)}")
        out.append(f"Keywords: {_keywords(block_text)}")
        out.append("")
    index_text = "\n".join(out).rstrip() + "\n"
    tmp = Path(session_dir) / "index.md.tmp"
    tmp.write_text(index_text, encoding="utf-8")
    os.replace(tmp, Path(session_dir) / "index.md")
    return index_text


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
        max_state_chars: int = 2400,
        max_recent_turns: int = 8,
        turn_user_chars: int = 200,
        turn_assistant_chars: int = 300,
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
        # its history replaced by a placeholder digest.)
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
        # Safety net: NEVER replace history with an empty digest — a pre-engine
        # session, a session whose recording failed, or one just reset must fall
        # back to native context (no placeholder digest).
        if not self._digest.strip():
            return None
        last_user_idx: Optional[int] = None
        for i, msg in enumerate(request_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = i
        if last_user_idx is None:
            return None
        # Leading system message(s) are always preserved.
        head: List[Dict[str, Any]] = []
        for msg in request_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                head.append(msg)
            else:
                break
        if last_user_idx <= 1:
            # First request of the session (system [+ one message]): nothing to
            # trim, BUT on a resumed session the loaded digest must still be
            # injected — otherwise prior context is silently lost (P3-10).
            if not self._digest.strip():
                return None
            digest_msg = {"role": "user", "content": DIGEST_PREFIX + self._digest.strip()}
            return [*head, digest_msg, *request_messages[last_user_idx:]]
        digest = self._digest.strip() or "(no prior context recorded for this session yet)"
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
        turn_no = self._turn_counter + 1
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
        tools_field = (
            f"Tools used: {', '.join(tool_list[:10])}\n" if tool_list else "")
        turn_digest = (
            f"### Turn {turn_no}\n"
            f"User: {_clip(user_text, self.turn_user_chars)}\n"
            f"{tools_field}"
            f"Assistant: {_clip(assistant_text, self.turn_assistant_chars) or '(no reply)'}\n"
        )
        self._digest = self._write_state(turn_digest)
        # Advance the counter only after BOTH files were written successfully.
        self._turn_counter = turn_no

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
        if not goal:
            # First recorded turn of the session: derive the goal from it.
            user_line = ""
            m = re.search(r"^User: (.*)$", new_turn_digest, re.MULTILINE)
            if m:
                user_line = m.group(1)
            goal = _clip(user_line, 300)
        # Keep only the most recent N turn sections from the previous digest.
        old_turns = _TURN_SECTION_RE.split(existing.split("## Recent turns", 1)[-1]) if "## Recent turns" in existing else []
        old_turns = [t for t in old_turns if t.strip()]
        # Guard the -0 slice bug: max_recent_turns=1 must keep NO old turns
        # (old_turns[-0:] would return the whole list).
        if self.max_recent_turns <= 1:
            kept: List[str] = []
        else:
            kept = old_turns[-(self.max_recent_turns - 1):]
        header = (
            "# Working state (markdown-memory)\n\n"
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
                    "Search archived earlier turns (recover details not in the working state)."
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

    @staticmethod
    def _cap_recall(sections: List[str]) -> List[str]:
        """Bound recall output: ~1500 chars per turn, ~4000 chars total."""
        capped: List[str] = []
        total = 0
        for sec in sections:
            clipped = sec[:1500]
            if capped and total + len(clipped) > 4000:
                break
            capped.append(clipped)
            total += len(clipped)
        return capped

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
            hits = self._cap_recall(sections[-limit:])
            return json.dumps({"matches": hits, "total_turns": len(sections)})
        terms = [t.lower() for t in re.split(r"\s+", query) if t]
        scored = []
        for pos, sec in enumerate(sections):
            low = sec.lower()
            score = sum(low.count(t) for t in terms)
            if score:
                scored.append((score, pos, sec))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        matches = [sec for _s, _p, sec in scored[:limit]]
        return json.dumps(
            {"matches": self._cap_recall(matches), "total_turns": len(sections)})

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
