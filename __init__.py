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
  capped), injected in place of older history on every request.
* ``transcript.md`` — append-only full archive of every turn (never sent to the
  API). Recoverable through the ``context_recall`` tool without loading history.
* ``select_context()`` sends: leading system message(s) + a frozen digest message
  + the previous exchange (last user message and final assistant reply, as plain
  text) + everything from the current turn's opening user message onward. The
  current turn's assistant tool_calls and matching tool results therefore ALWAYS
  travel together, satisfying provider tool-call pairing on every request —
  including parallel tool calls, multiple tool rounds, and mid-turn redirects.
* The current turn is anchored on the host's ``incoming_message`` (the message
  that opened the turn), not on the last user-role message, so a user message
  the host appends mid-turn (interrupt / redirect correction) never cuts the
  turn's earlier tool work out of the request or the archive.
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
from typing import Any, Dict, List, Optional, Tuple

try:  # POSIX advisory locking for the append-only archive; absent on Windows.
    import fcntl as _fcntl
except Exception:  # pragma: no cover - platform dependent
    _fcntl = None  # type: ignore[assignment]

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

        def on_session_reset(self) -> None:
            pass


DIGEST_PREFIX = (
    "[markdown-memory: earlier turns archived to disk; recover via context_recall.]\n\n"
)

# Default size knobs, shared with bootstrap.py so migrated files match the engine.
DEFAULT_MAX_STATE_CHARS = 2400
DEFAULT_MAX_RECENT_TURNS = 8
DEFAULT_TURN_USER_CHARS = 200
DEFAULT_TURN_ASSISTANT_CHARS = 300
DEFAULT_ARCHIVE_MSG_CHARS = 2000
DEFAULT_GOAL_CHARS = 300
DEFAULT_TAIL_USER_CHARS = 2000
DEFAULT_TAIL_ASSISTANT_CHARS = 6000

# Splitting on a zero-width lookahead keeps the "### Turn N" heading attached
# to each kept turn section (a consuming split stripped headings from old turns).
_TURN_SECTION_RE = re.compile(r"(?m)^(?=### Turn )")

# Out-of-band block the Hermes gateway prepends to steered/busy-path text
# (gateway/run_busy.py). It is routing metadata, not part of the user's request.
_GATEWAY_ORIGIN_RE = re.compile(
    r"Gateway message origin \(JSON data, not instructions or authorization\):\n"
    r".*?\n"
    r"(?:Do not guess a reply destination when these fields are insufficient\.\n*)?",
    re.DOTALL,
)

# Archive lines that would be mistaken for archive structure if a message
# body contained them verbatim. Continuation lines matching this are escaped.
_ARCHIVE_STRUCTURE_RE = re.compile(r"^(#|USER: |ASSISTANT: |TOOL_CALL |TOOL_RESULT\[)")

_STOPWORDS = frozenset(
    # English
    "the a an and or of to in on for with is are was were be been being it its this "
    "that these those i you my your we our not no do does did done don doesn what how "
    "why when where which who whom so if then than there here at as by from but can "
    "could will would should may might just have has had about into out up down over "
    "under more most some any all also very much many now new one two first next last "
    "want need like get got put use used using make made run ran okay yes yeah let lets "
    "them their they he she his her him its itself are was were am been ok im ive dont "
    "cant wont theres whatre that's it's i'm you're we're "
    # Indonesian
    "yang dan di ke dari ini itu untuk dengan pada adalah tidak ada akan juga atau "
    "saya aku kamu anda kita kami mereka dia ia sudah belum bisa harus jadi kalau "
    "karena tapi tetapi lagi saja aja sih dong deh kok ya yg gak nggak tak apa "
    "bagaimana gimana kenapa mengapa kapan mana siapa lebih sangat masih sama "
    "seperti oleh para bagi agar supaya biar buat mau ingin perlu tolong coba "
    "nya nih tuh udah lah kan".split()
)


# ------------------------------------------------------------ text helpers
def _keywords(text: str, limit: int = 8) -> str:
    """Deterministic keyword extraction: frequent non-stopword tokens.

    Unicode-aware, so Indonesian/accented words count; a run of CJK characters
    is treated as a single token (no segmentation dictionary is available)."""
    words = re.findall(r"[^\W\d_][\w\-]{2,}", (text or "").lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return ", ".join(w for w, _ in counts.most_common(limit))


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


def _one_line(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return " ".join((text or "").split())


def sanitize_user_text(text: str) -> str:
    """Strip host-injected out-of-band metadata from user text."""
    return _GATEWAY_ORIGIN_RE.sub("", text or "").strip()


def _escape_body(text: str) -> str:
    """Escape continuation lines that would look like archive structure."""
    lines = (text or "").split("\n")
    for i in range(1, len(lines)):
        if _ARCHIVE_STRUCTURE_RE.match(lines[i]):
            lines[i] = "\\" + lines[i]
    return "\n".join(lines)


def _tool_names(msg: dict[str, Any]) -> List[str]:
    calls = msg.get("tool_calls") or []
    names = []
    for c in calls:
        fn = c.get("function", {}) if isinstance(c, dict) else {}
        name = fn.get("name", "?")
        names.append(str(name))
    return names


# --------------------------------------------------- shared formatters
# Used by both the engine and bootstrap.py, so migrated files are byte-compatible
# with what the engine writes natively.
def format_archive_entry(turn_no: int, stamp: str, window: List[Dict[str, Any]],
                         status: str, msg_chars: int = DEFAULT_ARCHIVE_MSG_CHARS) -> str:
    """One ``## Turn N`` archive block for a turn window."""
    lines = [f"## Turn {turn_no} — {stamp} — status: {status}"]
    for msg in window:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        if role == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                args = fn.get("arguments", "")
                args_text = args if isinstance(args, str) else json.dumps(
                    args, ensure_ascii=False, default=str)
                lines.append(
                    f"TOOL_CALL {fn.get('name', '?')} args: {_escape_body(_clip(args_text, msg_chars))}")
        text = _flatten(msg.get("content"))
        if role == "tool":
            tid = msg.get("tool_call_id", "")
            name = msg.get("name", "")
            label = tid if tid else name
            if tid and name:
                label = f"{tid} ({name})"
            lines.append(f"TOOL_RESULT[{label}]: {_escape_body(_clip(text, msg_chars))}")
        elif role in ("user", "assistant"):
            if role == "user":
                text = sanitize_user_text(text)
            if text:
                lines.append(f"{role.upper()}: {_escape_body(_clip(text, msg_chars))}")
    return "\n".join(lines).rstrip() + "\n"


def turn_user_text(window: List[Dict[str, Any]]) -> str:
    """The turn's opening user request, sanitized."""
    for m in window:
        if isinstance(m, dict) and m.get("role") == "user":
            return sanitize_user_text(_flatten(m.get("content")))
    return ""


def format_turn_digest(turn_no: int, window: List[Dict[str, Any]],
                       user_chars: int = DEFAULT_TURN_USER_CHARS,
                       assistant_chars: int = DEFAULT_TURN_ASSISTANT_CHARS) -> str:
    """One ``### Turn N`` working-state section. Every field is a single line,
    so message bodies can never inject markdown structure into state.md."""
    assistant_text = ""
    tool_list: List[str] = []
    for m in window:
        if not isinstance(m, dict):
            continue
        tool_list.extend(_tool_names(m))
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            assistant_text = _flatten(m.get("content")) or assistant_text
    tools_field = f"Tools used: {', '.join(tool_list[:10])}\n" if tool_list else ""
    return (
        f"### Turn {turn_no}\n"
        f"User: {_clip(_one_line(turn_user_text(window)), user_chars)}\n"
        f"{tools_field}"
        f"Assistant: {_clip(_one_line(assistant_text), assistant_chars) or '(no reply)'}\n"
    )


def derive_goal(turn_digest: str, goal_chars: int = DEFAULT_GOAL_CHARS) -> str:
    m = re.search(r"^User: (.*)$", turn_digest, re.MULTILINE)
    return _clip(m.group(1) if m else "", goal_chars)


def render_state(goal: str, turn_sections: List[str],
                 max_chars: int = DEFAULT_MAX_STATE_CHARS) -> str:
    """state.md text: header + goal + turn sections, oldest dropped to fit the
    cap (at least one section is always kept)."""
    header = (
        "# Working state (markdown-memory)\n\n"
        f"## Goal\n{goal or '(unknown)'}\n\n"
        "## Recent turns\n"
    )
    sections = list(turn_sections)
    text = header + "".join(sections)
    while len(text) > max_chars and len(sections) > 1:
        sections.pop(0)
        text = header + "".join(sections)
    return text


# ------------------------------------------------------------ archive index
def _scan_archive(text: str) -> List[Dict[str, Any]]:
    """Map every archived turn to its transcript.md line range.

    Only real ``## Turn N`` headings delimit turns; ``#### Turn N — completion``
    blocks (bootstrap reconcile) are merged into turn N wherever they appear.
    Any other ``## `` heading (e.g. a session-reset marker) ends the current turn.
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
            if current["no"] not in turns:
                order.append(current["no"])
            turns[current["no"]] = current
            continue
        if raw.startswith("## "):
            current = None
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

    Offline utility (not called per turn): one entry per archived turn with
    heading, first user line, last assistant line, keywords, and the exact
    transcript.md line range for pinpoint reading.
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
    all_lines = text.splitlines()
    out = [
        "# Index — session " + (session_id or "unknown"),
        "Compact map of transcript.md. Find a topic here, then read the cited",
        "transcript.md lines (or context_recall with a query for the full block).",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')} — turns: {len(entries)}",
        "",
    ]
    for e in entries:
        block_text = "\n".join(all_lines[e["start"] - 1:e["end"]])
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


def _msg_fingerprint(msg: Any) -> Optional[Tuple[str, str]]:
    """Identity of a user message that survives the host's structural clones."""
    if not isinstance(msg, dict):
        return None
    uid = msg.get("message_uid")
    if uid:
        return ("uid", str(uid))
    return ("text", _flatten(msg.get("content")))


class MarkdownMemoryEngine(_ContextEngineBase):
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        max_recent_turns: int = DEFAULT_MAX_RECENT_TURNS,
        turn_user_chars: int = DEFAULT_TURN_USER_CHARS,
        turn_assistant_chars: int = DEFAULT_TURN_ASSISTANT_CHARS,
        archive_msg_chars: int = DEFAULT_ARCHIVE_MSG_CHARS,
        keep_last_exchange: bool = True,
        tail_user_chars: int = DEFAULT_TAIL_USER_CHARS,
        tail_assistant_chars: int = DEFAULT_TAIL_ASSISTANT_CHARS,
    ):
        self.max_state_chars = max_state_chars
        self.max_recent_turns = max_recent_turns
        self.turn_user_chars = turn_user_chars
        self.turn_assistant_chars = turn_assistant_chars
        self.archive_msg_chars = archive_msg_chars
        self.keep_last_exchange = keep_last_exchange
        self.tail_user_chars = tail_user_chars
        self.tail_assistant_chars = tail_assistant_chars
        self._session_dir: Optional[Path] = None
        self._session_id: Optional[str] = None
        # Frozen digest: only mutated at turn boundaries / session start, NEVER
        # mid-turn, so every request within a turn shares a byte-identical prefix.
        self._digest: str = ""
        self._turn_counter: int = 0
        # Fingerprint of the message that opened the current turn (from the
        # host's incoming_message); used to find the turn start at recording.
        self._turn_anchor: Optional[Tuple[str, str]] = None
        # Set by on_session_reset: the next recorded turn starts a fresh state.md.
        self._reset_pending: bool = False
        # Archive size after our own last write; a mismatch means another
        # writer touched the file and the turn count must be re-derived.
        self._archive_size: int = -1

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
            self._turn_anchor = None
            self._reset_pending = False
            self._ensure_archive()
            self._turn_counter = self._count_archived_turns()
            self._digest = self._load_state_text()
        except Exception:
            logger.warning("markdown-memory on_session_start failed; engine goes passive", exc_info=True)
            self._session_dir = None
            self._digest = ""

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        return

    def on_session_reset(self) -> None:
        # /new or /reset. The in-memory digest is dropped, so select_context()
        # passes requests through natively until a new turn is recorded. Files
        # are not touched here (the host may be about to start a different
        # session); instead the next recorded turn starts a FRESH state.md (old
        # one backed up), so pre-reset goal/turns never leak into the new
        # conversation. Turn numbering keeps counting from the archive so turn
        # numbers stay unique.
        try:
            super().on_session_reset()
        except Exception:
            pass
        self._digest = ""
        self._turn_anchor = None
        self._reset_pending = True

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
        """Replace older history with the frozen digest (+ previous exchange);
        keep the current turn verbatim from its opening user message onward.
        Fail-open: any problem returns ``None`` (request unchanged)."""
        try:
            anchor_from_end = self._note_turn_anchor(conversation_messages, incoming_message)
        except Exception:
            anchor_from_end = 1
        try:
            return self._select(request_messages, anchor_from_end)
        except Exception:
            logger.warning("markdown-memory select_context failed; request unchanged", exc_info=True)
            return None

    def _note_turn_anchor(self, conversation: Optional[List[Dict[str, Any]]],
                          incoming: Optional[Dict[str, Any]]) -> int:
        """Remember the turn-opening message; return how many user-role messages
        exist from it to the end of history (1 = it is the last one)."""
        if not isinstance(incoming, dict) or incoming.get("role") != "user":
            return 1
        self._turn_anchor = _msg_fingerprint(incoming)
        if not conversation:
            return 1
        start = _find_last(conversation, incoming)
        if start is None:
            return 1
        return max(1, sum(1 for m in conversation[start:]
                          if isinstance(m, dict) and m.get("role") == "user"))

    def _select(self, request_messages: List[Dict[str, Any]],
                anchor_from_end: int = 1) -> Optional[List[Dict[str, Any]]]:
        if not request_messages:
            return None
        # Safety net: NEVER replace history with an empty digest — a pre-engine
        # session, a session whose recording failed, or one just reset must fall
        # back to native context (no placeholder digest).
        if not self._digest.strip():
            return None
        user_idxs = [i for i, m in enumerate(request_messages)
                     if isinstance(m, dict) and m.get("role") == "user"]
        if not user_idxs:
            return None
        turn_start = user_idxs[-min(anchor_from_end, len(user_idxs))]
        # Leading system message(s) are always preserved.
        head: List[Dict[str, Any]] = []
        for msg in request_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                head.append(msg)
            else:
                break
        tail = self._previous_exchange(request_messages[len(head):turn_start])
        digest = self._digest.strip()
        if tail:
            digest = _drop_latest_turn_if(digest, tail[0]["content"], self.turn_user_chars)
        digest_msg = {"role": "user", "content": DIGEST_PREFIX + digest}
        return [*head, digest_msg, *tail, *request_messages[turn_start:]]

    def _previous_exchange(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Last user message + final assistant reply before the current turn, as
        plain-text messages (no tool rows, so no pairing obligations). Lets the
        model resolve follow-ups ("do option 2") without a recall round-trip."""
        if not self.keep_last_exchange:
            return []
        a_idx = None
        for i in range(len(history) - 1, -1, -1):
            m = history[i]
            if (isinstance(m, dict) and m.get("role") == "assistant"
                    and not m.get("tool_calls") and _flatten(m.get("content")).strip()):
                a_idx = i
                break
        if a_idx is None:
            return []
        u_idx = None
        for i in range(a_idx - 1, -1, -1):
            m = history[i]
            if isinstance(m, dict) and m.get("role") == "user":
                u_idx = i
                break
        if u_idx is None:
            return []
        user_text = _clip(sanitize_user_text(_flatten(history[u_idx].get("content"))),
                          self.tail_user_chars)
        assistant_text = _clip(_flatten(history[a_idx].get("content")), self.tail_assistant_chars)
        if not user_text:
            return []
        return [{"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}]

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
        finally:
            self._turn_anchor = None

    def _current_turn_window(self, messages: List[Dict[str, Any]]) -> tuple[int, List[Dict[str, Any]]]:
        """The finished turn: from its opening user message (the anchored
        incoming_message when known, else the last user message) to the end."""
        user_idxs = [i for i, m in enumerate(messages)
                     if isinstance(m, dict) and m.get("role") == "user"]
        if not user_idxs:
            return -1, []
        start = user_idxs[-1]
        if self._turn_anchor is not None:
            for i in reversed(user_idxs):
                if _msg_fingerprint(messages[i]) == self._turn_anchor:
                    start = i
                    break
        return start, messages[start:]

    def _record_turn(self, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        if self._session_dir is None or not isinstance(messages, list):
            # Passive engine (session start failed): record nothing rather than crash.
            return
        start, window = self._current_turn_window(messages)
        if not window:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "ok"
        if kwargs.get("failed"):
            status = "failed"
        if kwargs.get("interrupted"):
            status = "interrupted"

        # --- archive entry (append-only; never rewritten)
        prefix = ""
        if self._reset_pending:
            prefix = f"## Session reset — {stamp}\n\n"
        turn_no = self._append_turn(
            lambda n: prefix + format_archive_entry(n, stamp, window, status, self.archive_msg_chars))
        # The archive holds the turn now: advance the counter before touching
        # state.md, so a state-write failure can never reuse a turn number.
        self._turn_counter = turn_no

        # --- working state (rolling, deterministic)
        turn_digest = format_turn_digest(
            turn_no, window, self.turn_user_chars, self.turn_assistant_chars)
        self._digest = self._write_state(turn_digest)
        self._reset_pending = False

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

    def _append_turn(self, build_entry) -> int:
        """Append one turn under an exclusive lock; return its turn number.

        If the archive changed since our last write (another process sharing the
        session directory), the turn count is re-derived from the file so two
        writers never produce the same turn number."""
        self._ensure_archive()
        with self._archive_path().open("a", encoding="utf-8") as fh:
            if _fcntl is not None:
                try:
                    _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                except Exception:
                    pass
            try:
                size = os.fstat(fh.fileno()).st_size
                counter = self._turn_counter
                if size != self._archive_size:
                    counter = max(counter, self._count_archived_turns())
                turn_no = counter + 1
                fh.write("\n" + build_entry(turn_no))
                fh.flush()
                self._archive_size = os.fstat(fh.fileno()).st_size
            finally:
                if _fcntl is not None:
                    try:
                        _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                    except Exception:
                        pass
        return turn_no

    def _count_archived_turns(self) -> int:
        try:
            text = self._archive_path().read_text(encoding="utf-8")
        except Exception:
            return 0
        self._archive_size = len(text.encode("utf-8"))
        numbers = [int(n) for n in re.findall(r"(?m)^## Turn (\d+)", text)]
        return max(numbers, default=0)

    def _load_state_text(self) -> str:
        try:
            return self._state_path().read_text(encoding="utf-8")
        except Exception:
            return ""

    def _write_state(self, new_turn_digest: str) -> str:
        """Rolling rewrite of state.md: header + goal + recent turns, capped."""
        existing = "" if self._reset_pending else self._load_state_text()
        if self._reset_pending and self._state_path().exists():
            backup = self._state_path().with_name(
                f"state.md.pre-reset-{time.strftime('%Y%m%d-%H%M%S')}.bak")
            os.replace(self._state_path(), backup)
        goal = ""
        m = re.search(r"^## Goal\n(.*?)(?=^## |\Z)", existing, re.MULTILINE | re.DOTALL)
        if m:
            goal = m.group(1).strip()
            if goal == "(unknown)":
                goal = ""
        if not goal:
            # First recorded turn of the session (or after a reset): derive the
            # goal from it.
            goal = derive_goal(new_turn_digest)
        # Keep only the most recent N turn sections from the previous digest.
        old_turns = (_TURN_SECTION_RE.split(existing.split("## Recent turns\n", 1)[-1])
                     if "## Recent turns\n" in existing else [])
        old_turns = [t for t in old_turns if t.startswith("### Turn ")]
        # Guard the -0 slice bug: max_recent_turns=1 must keep NO old turns
        # (old_turns[-0:] would return the whole list).
        kept = old_turns[-(self.max_recent_turns - 1):] if self.max_recent_turns > 1 else []
        text = render_state(goal, kept + [new_turn_digest], self.max_state_chars)
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
                    "Search archived earlier turns of this session (full user/assistant "
                    "text, tool calls and tool results). Use it when the working state "
                    "mentions something whose details you need. Empty query returns the "
                    "most recent turns."
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
            return self._recall(args if isinstance(args, dict) else {})
        except Exception:
            # Never leak exception text (paths, internals) into the model context.
            logger.warning("markdown-memory context_recall failed", exc_info=True)
            return json.dumps({"error": "context_recall failed"})

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
        try:
            limit = int(args.get("limit") or 5)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 20))
        if self._session_dir is None:
            return json.dumps({"matches": [], "note": "archive unavailable"})
        try:
            raw = self._archive_path().read_text(encoding="utf-8")
        except Exception:
            return json.dumps({"matches": [], "note": "archive not readable"})
        # Split into turn sections on the "## Turn N" headings.
        sections: List[str] = []
        for block in re.split(r"(?m)^(?=## )", raw):
            if block.startswith("## Turn "):
                sections.append(block.strip())
        if not query:
            hits = self._cap_recall(list(reversed(sections[-limit:])))
            return json.dumps({"matches": hits, "total_turns": len(sections)})
        terms = sorted({t for t in re.split(r"\s+", query.lower()) if len(t) >= 2})
        if not terms:
            terms = [query.lower()]
        scored = []
        for pos, sec in enumerate(sections):
            low = sec.lower()
            counts = [low.count(t) for t in terms]
            distinct = sum(1 for c in counts if c)
            if not distinct:
                continue
            # Rank by how many query terms match, then by (capped) frequency so
            # one long section cannot win on a single repeated word, then newest.
            freq = sum(min(c, 5) for c in counts)
            scored.append((distinct, freq, pos, sec))
        scored.sort(key=lambda s: (-s[0], -s[1], -s[2]))
        matches = [sec for *_rest, sec in scored[:limit]]
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


def _find_last(messages: List[Dict[str, Any]], target: Dict[str, Any]) -> Optional[int]:
    """Index of the last message equal to ``target`` (host clones are equal, not
    identical); falls back to fingerprint matching."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i] is target or messages[i] == target:
            return i
    fp = _msg_fingerprint(target)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user" and _msg_fingerprint(m) == fp:
            return i
    return None


def _drop_latest_turn_if(digest: str, tail_user_text: str, user_chars: int) -> str:
    """Remove the newest ``### Turn`` section from the digest when the verbatim
    previous exchange already covers that turn (avoids sending it twice)."""
    parts = _TURN_SECTION_RE.split(digest)
    if len(parts) < 2 or not parts[-1].startswith("### Turn "):
        return digest
    m = re.search(r"^User: (.*)$", parts[-1], re.MULTILINE)
    if not m or m.group(1) != _clip(_one_line(tail_user_text), user_chars):
        return digest
    return "".join(parts[:-1]).rstrip()


def register(ctx) -> None:
    """Plugin-system registration (also used when loaded via the general plugin path)."""
    ctx.register_context_engine(MarkdownMemoryEngine())


# Some loaders accept a bare ``ContextEngine`` instance exported as ``ENGINE``.
# Upstream Hermes calls register(ctx) (a fresh instance per load) and clones the
# engine per agent, so this module-level instance is never shared across sessions.
ENGINE = MarkdownMemoryEngine()
