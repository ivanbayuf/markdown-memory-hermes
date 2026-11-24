"""Bootstrap markdown-memory from the native session database — READ-ONLY.

Reads existing conversation history from ``$HERMES_HOME/state.db`` and writes
markdown-memory state/transcript files for sessions currently referenced by
``gateway_routing``. NEVER writes to the database (opened with SQLite
``mode=ro``) and NEVER touches native session records.

Safety properties:
* ``--dry-run`` is the DEFAULT: zero memory-file writes, report only.
* ``--write`` (explicit) creates files atomically (tmp + os.replace), refuses
  to overwrite any existing state.md/transcript.md, and is idempotent.
* Tool calls/results (including parallel calls), ``tool_call_id`` pairing and
  original ordering are preserved verbatim in the archive.
* Turn numbering continues from the migrated count (the engine's
  ``on_session_start`` counts archived ``## Turn`` headings).
* Ambiguous history (orphaned tool results, no user messages, first row not
  user, empty sessions) is FLAGGED, not silently migrated.
* Compact summaries / compacted markers inside history are preserved as-is.

Usage:
    python3 bootstrap.py                 # dry-run report (no writes)
    python3 bootstrap.py --write         # create memory files (explicit)
    python3 bootstrap.py --write --session <id> [--session <id> ...]
    python3 bootstrap.py --reconcile --backfill --write [--session <id> ...]
    python3 bootstrap.py --repair-state --write [--session <id> ...]
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the plugin's extraction/formatting so migrated files are byte-compatible
# with what the engine itself would have written.
import importlib.util

_PLUGIN_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "markdown_memory_bootstrap_import", _PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(_PLUGIN_DIR)])
_mm = importlib.util.module_from_spec(_spec)
sys.modules["markdown_memory_bootstrap_import"] = _mm
_spec.loader.exec_module(_mm)

DIGEST_PREFIX = _mm.DIGEST_PREFIX
_clip = _mm._clip
_flatten = _mm._flatten
_tool_names = _mm._tool_names

STALE_NOTE = "stale — not migrated (not referenced by gateway_routing)"


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    """Strictly read-only SQLite connection (URI mode=ro; never creates files)."""
    if not db_path.exists():
        raise FileNotFoundError(f"session database not found: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _target_sessions(con: sqlite3.Connection) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split sessions into (targets from gateway_routing, stale others)."""
    targets: Dict[str, Dict[str, Any]] = {}
    try:
        rows = con.execute("SELECT session_key, entry_json FROM gateway_routing").fetchall()
    except sqlite3.Error:
        rows = []
    for key, entry_json in rows:
        try:
            entry = json.loads(entry_json or "{}")
        except Exception:
            continue
        sid = entry.get("session_id")
        if not sid or sid in targets:
            continue
        scope = key.split(":", 2)[0] if key else ""
        profile = key.split(":")[1] if key and key.count(":") >= 1 else ""
        targets[sid] = {"session_id": sid, "profile": profile, "session_key": key,
                        "source": "gateway_routing"}
    stale: List[Dict[str, Any]] = []
    sess_rows = con.execute(
        "SELECT id, source, profile_name, message_count, input_tokens, archived,"
        " title, datetime(last_activity_at, 'unixepoch') FROM sessions"
        " ORDER BY last_activity_at DESC").fetchall()
    for (sid, source, profile, mcount, intok, archived, title, last_act) in sess_rows:
        if sid in targets:
            targets[sid].update({
                "source_kind": source, "profile_name": profile, "message_count_db": mcount,
                "input_tokens": intok, "archived_flag": archived, "title": title,
                "last_activity": last_act,
            })
        else:
            stale.append({"session_id": sid, "source": source, "profile": profile,
                          "message_count": mcount, "input_tokens": intok,
                          "archived": archived, "title": title, "last_activity": last_act})
    ordered = []
    for sid, info in targets.items():
        sess = con.execute(
            "SELECT source, profile_name, message_count, input_tokens, archived, title,"
            " datetime(last_activity_at, 'unixepoch') FROM sessions WHERE id=?",
            (sid,)).fetchone()
        if sess is None:
            info["status"] = "missing"
            info["reason"] = "gateway_routing references a session not present in the sessions table"
            ordered.append(info)
            continue
        info.update({"source_kind": sess[0], "profile_name": sess[1], "message_count_db": sess[2],
                     "input_tokens": sess[3], "archived_flag": sess[4], "title": sess[5],
                     "last_activity": sess[6]})
        if sess[4]:  # archived
            info["status"] = "archived"
            info["reason"] = "session is archived in the DB; nothing to migrate"
        ordered.append(info)
    return ordered, stale


def _load_active_messages(con: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    rows = con.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name, timestamp,"
        " _compressed_summary, compacted"
        " FROM messages WHERE session_id=? AND active=1 ORDER BY id",
        (session_id,)).fetchall()
    msgs = []
    for (role, content, tool_call_id, tool_calls, tool_name, ts, summary, compacted) in rows:
        m: Dict[str, Any] = {"role": role, "content": content, "timestamp": ts}
        if tool_call_id:
            m["tool_call_id"] = tool_call_id
        if tool_name:
            m["name"] = tool_name
        if tool_calls:
            try:
                parsed = json.loads(tool_calls)
            except Exception:
                m["_ambiguous_tool_calls"] = True
                parsed = []
            # Normalise to the wire shape the engine/archive expect.
            norm = []
            for c in parsed if isinstance(parsed, list) else []:
                fn = c.get("function", {}) if isinstance(c, dict) else {}
                cid = c.get("id") or c.get("call_id") or ""
                norm.append({"id": cid, "type": "function",
                             "function": {"name": fn.get("name", "?"),
                                          "arguments": fn.get("arguments", "")}})
            m["tool_calls"] = norm
        if summary:
            m["_compressed_summary"] = 1
        if compacted:
            m["_compacted"] = 1
        msgs.append(m)
    return msgs


def _group_turns(msgs: List[Dict[str, Any]]
                 ) -> Tuple[List[List[Dict[str, Any]]], List[str], List[str]]:
    """Group into turns at user-message boundaries; collect ambiguity flags
    and notes. Returns (turns, flags, notes)."""
    flags: List[str] = []
    idx = 0
    # session_meta rows are metadata, not conversation content: skip wherever they appear.
    leading_meta = 0
    while idx < len(msgs) and msgs[idx].get("role") in ("session_meta",):
        idx += 1
        leading_meta += 1
    if leading_meta:
        flags.append(f"leading session_meta rows skipped ({leading_meta})")
    if idx >= len(msgs):
        return [], flags, []
    if msgs[idx].get("role") != "user":
        flags.append(
            f"first active conversation row is role={msgs[idx].get('role')!r}, not user")
    turns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for m in msgs[idx:]:
        role = m.get("role")
        if role == "session_meta":
            continue
        if role == "user":
            if current:
                turns.append(current)
            current = [m]
            continue
        if not current:
            flags.append(f"conversation content before first user message: role={role}")
            current = [m]  # keep it in a synthetic leading turn for the archive
            continue
        current.append(m)
    if current:
        turns.append(current)
    # Orphaned tool results: tool rows whose tool_call_id was never declared by an
    # earlier assistant row within the same turn (or any earlier turn on record).
    declared: set = set()
    orphaned = 0
    for turn in turns:
        for m in turn:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                declared.update(c.get("id", "") for c in m["tool_calls"])
            elif m.get("role") == "tool":
                if m.get("tool_call_id") not in declared:
                    orphaned += 1
                else:
                    declared.discard(m.get("tool_call_id"))
    if orphaned:
        flags.append(f"{orphaned} orphaned tool result(s) (tool_call_id not declared earlier)")
    if not any(any(m.get("role") == "user" for m in t) for t in turns):
        flags.append("no user messages found in active history")
    n_user_turns = sum(1 for t in turns if t and t[0].get("role") == "user")
    if len(turns) != n_user_turns:
        flags.append(
            f"turn boundary mismatch: {len(turns)} grouped turns but {n_user_turns} user-led")
    compacted_rows = sum(1 for m in msgs if m.get("_compacted") or m.get("_compressed_summary"))
    notes = []
    if compacted_rows:
        # Not an ambiguity: compacted rows are what native replay sends; keep as-is.
        notes.append(f"history contains {compacted_rows} compacted/summary row(s) — preserved as-is")
    return turns, flags, notes


def _db_turns(con: sqlite3.Connection, session_id: str) -> List[List[Tuple]]:
    """Return active, user-led DB windows in native row order."""
    rows = con.execute(
        "SELECT id, role, tool_calls, tool_call_id, content, tool_name, timestamp"
        " FROM messages WHERE session_id=? AND active=1 ORDER BY id", (session_id,)
    ).fetchall()
    turns: List[List[Tuple]] = []
    current: List[Tuple] = []
    for row in rows:
        if row[1] == "session_meta":
            continue
        if row[1] == "user":
            if current:
                turns.append(current)
            current = [row]
        elif current:
            current.append(row)
    if current:
        turns.append(current)
    return turns


def _row_to_message(row: Tuple) -> Dict[str, Any]:
    """Convert the _db_turns row shape to the engine/archive message shape."""
    _rid, role, tool_calls, tool_call_id, content, tool_name, timestamp = row
    msg: Dict[str, Any] = {"role": role, "content": content, "timestamp": timestamp}
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    if tool_name:
        msg["name"] = tool_name
    if tool_calls:
        try:
            parsed = json.loads(tool_calls)
        except Exception:
            parsed = []
        calls = []
        for call in parsed if isinstance(parsed, list) else []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            calls.append({
                "id": call.get("id", "") if isinstance(call, dict) else "",
                "type": "function",
                "function": {"name": fn.get("name", "?"),
                             "arguments": fn.get("arguments", "")},
            })
        msg["tool_calls"] = calls
    return msg


def _turn_is_closed(window: List[Tuple], now: Optional[float] = None) -> bool:
    """A turn is closed only after a non-empty final assistant row ages out."""
    if not window or window[-1][1] != "assistant" or window[-1][2]:
        return False
    final_text = _flatten(window[-1][4]).strip()
    if not final_text:
        return False
    timestamp = window[-1][6]
    if timestamp is None:
        return False
    return (time.time() if now is None else now) - float(timestamp) > 120


def _archive_block_info(text: str) -> List[Tuple[int, int, str, str]]:
    """Return (number, offset, block, first user line) for real ## headings."""
    headings = [(int(m.group(1)), m.start())
                for m in re.finditer(r"(?m)^## Turn (\d+)", text)]
    result = []
    for i, (number, start) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else len(text)
        block = text[start:end]
        user = re.search(r"(?m)^USER: (.*)$", block)
        result.append((number, start, block, user.group(1) if user else ""))
    return result


def _db_user_first_line(window: List[Tuple]) -> str:
    for row in window:
        if row[1] == "user":
            value = _clip(_flatten(row[4]), 2000)
            return value.split("\n", 1)[0]
    return ""


def reconcile_session(session_id: str, con: sqlite3.Connection,
                      hermes_home: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Append missing rows for archived turns that were partial at snapshot time.

    Completion blocks are appended to transcript.md as ``#### Turn N — completion``
    — the ``####`` prefix deliberately does NOT match the engine's ``^## Turn ``
    turn counter, so numbering is unchanged. Idempotent: a turn is reconciled at
    most once (marker check). Database access is strictly read-only, and the
    last DB turn (still live) is never reconciled — the engine owns it.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:120]
    mem_dir = hermes_home / "context-memory" / safe
    transcript_path = mem_dir / "transcript.md"
    result: Dict[str, Any] = {"session_id": session_id,
                              "mode": "dry-run" if dry_run else "write"}
    if not transcript_path.exists():
        result["status"] = "no transcript"
        return result
    text = transcript_path.read_text(encoding="utf-8")
    blocks = _archive_block_info(text)
    if not blocks:
        result["status"] = "no turns"
        return result
    # Build DB turn windows (user-led) from the read-only connection.
    db_turns = _db_turns(con, session_id)
    result["archived_turns"] = len(blocks)
    result["db_turns"] = len(db_turns)
    appended: List[int] = []
    already: List[int] = []
    additions: List[str] = []
    # Match each archive block to a DB turn by first-line USER text; repeated
    # texts are consumed in DB order (occurrence count), never by position.
    unmatched: Dict[str, List[int]] = {}
    for j, window in enumerate(db_turns):
        unmatched.setdefault(_db_user_first_line(window), []).append(j)
    for i, (tno, _start, _block, _user) in enumerate(blocks):
        candidates = unmatched.get(_user)
        if not candidates:
            continue
        j = candidates.pop(0)
        if re.search(rf"(?m)^#### Turn {tno} — completion", text):
            already.append(tno)
            continue
        window = db_turns[j]
        # Compare the archived block against the (complete) DB window and append
        # only rows missing from the archive (bootstrap snapshot race).
        end = blocks[i + 1][1] if i + 1 < len(blocks) else len(text)
        block = text[_start:end]
        archived_result_ids = set(re.findall(r"(?m)^TOOL_RESULT\[([^\] ]+)", block))
        # Archived text rows may wrap (clip keeps embedded newlines): compare on
        # the first line of each row, which is byte-stable for both sources.
        archived_texts = set(re.findall(r"(?m)^(?:ASSISTANT|USER): (.*)$", block))
        missing_rows: List[Tuple] = []
        for row in window:
            (_rid, role, tool_calls, tool_call_id, content, _tool_name, _ts) = row
            if role == "tool" and tool_call_id and tool_call_id not in archived_result_ids:
                missing_rows.append(row)
            elif role in ("user", "assistant"):
                t = _clip(_flatten(content), 2000)
                first_line = t.split("\n", 1)[0]
                if t and first_line not in archived_texts:
                    missing_rows.append(row)
        # Only CLOSED turns are reconciled; the last DB turn may still be live.
        if missing_rows and j < len(db_turns) - 1:
            additions.append(
                f"#### Turn {tno} — completion (reconciled {time.strftime('%Y-%m-%d %H:%M:%S')})")
            for row in missing_rows:
                (_rid, role, tool_calls, tool_call_id, content, _tool_name, _ts) = row
                if role == "tool" and tool_call_id:
                    additions.append(
                        f"TOOL_RESULT[{tool_call_id}]: {_clip(_flatten(content), 2000)}")
                elif role in ("user", "assistant"):
                    t = _flatten(content)
                    if t:
                        additions.append(f"{role.upper()}: {_clip(t, 2000)}")
            appended.append(tno)
        elif missing_rows and j == len(db_turns) - 1:
            result["live_turn_note"] = (
                f"turn {tno}: missing rows belong to the LIVE turn; left to the engine")
    if not dry_run and additions:
        with transcript_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + "\n".join(additions) + "\n")
    result["reconciled_turns"] = appended
    result["already_reconciled"] = already
    result["status"] = "ok"
    return result


def backfill_session(session_id: str, con: sqlite3.Connection,
                     hermes_home: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Append whole closed DB turns absent from the archive as real turns."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:120]
    transcript_path = hermes_home / "context-memory" / safe / "transcript.md"
    result: Dict[str, Any] = {"session_id": session_id,
                              "mode": "dry-run" if dry_run else "write"}
    if not transcript_path.exists():
        result.update(status="no transcript", backfilled_turns=[])
        return result
    text = transcript_path.read_text(encoding="utf-8")
    blocks = _archive_block_info(text)
    db_turns = _db_turns(con, session_id)
    result.update(archived_turns=len(blocks), db_turns=len(db_turns))
    archived_users = Counter(user for _n, _s, _b, user in blocks)
    missing: List[List[Tuple]] = []
    for window in db_turns:
        user = _db_user_first_line(window)
        if (_turn_is_closed(window) and archived_users.get(user, 0) == 0):
            missing.append(window)
        elif archived_users.get(user, 0) > 0:
            # Match each DB turn to one archived block. Counts matter when a
            # user repeats the same first line in multiple turns.
            archived_users[user] -= 1
    first_number = max((n for n, _s, _b, _u in blocks), default=0) + 1
    additions: List[str] = []
    appended: List[int] = []
    for offset, window in enumerate(missing):
        number = first_number + offset
        messages = [_row_to_message(row) for row in window]
        stamp_value = window[0][6]
        stamp = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(stamp_value)))
                 if stamp_value else time.strftime("%Y-%m-%d %H:%M:%S"))
        entry, _digest = _archive_entry(number, stamp, messages, "backfilled")
        additions.append(entry)
        appended.append(number)
    if not dry_run and additions:
        with transcript_path.open("a", encoding="utf-8") as fh:
            # Same layout as the engine: each entry preceded by a blank line.
            fh.write("".join("\n" + entry for entry in additions))
    result["backfilled_turns"] = appended
    result["would_append"] = len(additions)
    result["status"] = "ok"
    return result


def _parse_goal(state_path: Path) -> Optional[str]:
    if not state_path.exists():
        return None
    text = state_path.read_text(encoding="utf-8")
    match = re.search(r"^## Goal\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if not match:
        return None
    goal = match.group(1).strip()
    if not goal or goal == "(unknown)":
        return None
    return goal


def _state_with_goal(state_md: str, goal: str, max_chars: int = 2400) -> str:
    """Replace the generated goal while retaining _build_files' rolling/cap rules."""
    match = re.search(r"^(## Goal\n).*?(?=^## Recent turns\n)", state_md,
                      re.MULTILINE | re.DOTALL)
    if not match:
        return state_md
    updated = state_md[:match.start()] + match.group(1) + goal + "\n\n" + state_md[match.end():]
    # A preserved goal can be longer than the derived, clipped goal. Apply the
    # same oldest-turn removal policy used by the engine until the cap is met.
    while len(updated) > max_chars:
        # Headings may be timestamped ("### Turn N (stamp)") on legacy state
        # files, so match any suffix after the turn number.
        turns = list(re.finditer(r"^### Turn \d+.*\n", updated, re.MULTILINE))
        if len(turns) <= 1:
            break
        start = turns[0].start()
        end = turns[1].start() if len(turns) > 1 else len(updated)
        updated = updated[:start] + updated[end:]
    return updated


def _atomic_repair_backup(state_path: Path) -> Path:
    candidate = state_path.with_name(state_path.name + ".pre-repair.bak")
    if candidate.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = state_path.with_name(state_path.name + f".pre-repair-{stamp}.bak")
        index = 2
        while candidate.exists():
            candidate = state_path.with_name(
                state_path.name + f".pre-repair-{stamp}-{index}.bak")
            index += 1
    tmp = candidate.with_name(candidate.name + ".tmp-bootstrap")
    shutil.copy2(state_path, tmp)
    os.replace(tmp, candidate)
    return candidate


def repair_state_session(session_id: str, con: sqlite3.Connection,
                         hermes_home: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Rebuild state.md from closed DB turns without touching transcript.md."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:120]
    mem_dir = hermes_home / "context-memory" / safe
    state_path = mem_dir / "state.md"
    transcript_path = mem_dir / "transcript.md"
    result: Dict[str, Any] = {"session_id": session_id,
                              "mode": "dry-run" if dry_run else "write"}
    db_turns = _db_turns(con, session_id)
    closed = [turn for turn in db_turns if _turn_is_closed(turn)]
    result["closed_turns"] = len(closed)
    summaries = []
    for number, window in enumerate(closed, start=1):
        user = _db_user_first_line(window)
        assistant = ""
        for row in window:
            if row[1] == "assistant" and not row[2]:
                assistant = _flatten(row[4]).strip() or assistant
        summaries.append({"turn": number, "user": user, "assistant": assistant.split("\n", 1)[0]})
    result["repair_turns"] = summaries[-8:]
    if not state_path.exists():
        result.update(status="no state", migration_status="no state.md to repair")
        return result
    if not closed:
        result.update(status="nothing to repair",
                      migration_status="nothing to repair (no closed DB turns); state.md untouched")
        return result
    if not dry_run:
        goal = _parse_goal(state_path)
        # _build_files supplies the engine-compatible extraction, numbering,
        # recent-turn rolling, and 2400-character cap.
        windows = [[_row_to_message(row) for row in turn] for turn in closed]
        state_md, _unused_transcript, derived_goal, _count = _build_files(session_id, windows)
        state_md = _state_with_goal(state_md, goal or derived_goal, 2400)
        if state_path.read_text(encoding="utf-8") == state_md:
            result["migration_status"] = "state.md already repaired (idempotent no-op)"
            result["status"] = "ok"
            return result
        backup = _atomic_repair_backup(state_path)
        tmp = state_path.with_suffix(state_path.suffix + ".tmp-repair")
        tmp.write_text(state_md, encoding="utf-8")
        os.replace(tmp, state_path)
        result["backup"] = str(backup)
        result["migration_status"] = "state.md repaired"
    else:
        result["migration_status"] = "dry-run — no files written"
    result["status"] = "ok"
    return result


def _archive_entry(turn_no: int, stamp: str, window: List[Dict[str, Any]],
                   status: str) -> Tuple[str, str]:
    """Same archive/one-line-digest format the engine writes natively."""
    lines = [f"## Turn {turn_no} — {stamp} — status: {status}"]
    for msg in window:
        role = msg.get("role", "?")
        if role == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                args = fn.get("arguments", "")
                args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
                lines.append(f"TOOL_CALL {fn.get('name', '?')} args: {_clip(args_text, 2000)}")
        text = _flatten(msg.get("content"))
        if role == "tool":
            tid = msg.get("tool_call_id", "")
            name = msg.get("name", "")
            label = tid if tid else name
            if tid and name:
                label = f"{tid} ({name})"
            lines.append(f"TOOL_RESULT[{label}]: {_clip(text, 2000)}")
        elif role in ("user", "assistant"):
            if text:
                lines.append(f"{role.upper()}: {_clip(text, 2000)}")
    entry = "\n".join(lines).rstrip() + "\n"
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
        f"User: {_clip(user_text, 200)}\n"
        f"{tools_field}"
        f"Assistant: {_clip(assistant_text, 300) or '(no reply)'}\n"
    )
    return entry, turn_digest


def _build_files(session_id: str, turns: List[List[Dict[str, Any]]],
                 max_state_chars: int = 2400, max_recent_turns: int = 8
                 ) -> Tuple[str, str, str, int]:
    """Return (state_md, transcript_md, goal, turn_count)."""
    transcript = [f"# Transcript archive — session {session_id}\n"
                  "Append-only. Never sent to the API; search it with context_recall.\n"
                  f"Bootstrapped from native session history (read-only DB replay), {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"]
    digests: List[str] = []
    goal = ""
    for no, window in enumerate(turns, start=1):
        ts = window[0].get("timestamp") or 0
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else time.strftime("%Y-%m-%d %H:%M:%S")
        entry, turn_digest = _archive_entry(no, stamp, window, "bootstrapped")
        transcript.append("\n" + entry)
        digests.append(turn_digest)
        if not goal:
            m = re.search(r"^User: (.*)$", turn_digest, re.MULTILINE)
            goal = _clip(m.group(1) if m else "", 300)
    header = ("# Working state (markdown-memory)\n\n"
              f"## Goal\n{goal or '(unknown)'}\n\n"
              "## Recent turns\n")
    kept = digests[-max_recent_turns:]
    text = header + "".join(kept)
    while len(text) > max_state_chars and len(kept) > 1:
        kept.pop(0)
        text = header + "".join(kept)
    return text, "".join(transcript), goal, len(turns)


def _atomic_write(path: Path, content: str) -> None:
    """Create ``path`` atomically; refuse to overwrite an existing file."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp-bootstrap")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def bootstrap_session(session_row: Dict[str, Any], con: sqlite3.Connection,
                      hermes_home: Path, write: bool) -> Dict[str, Any]:
    sid = session_row["session_id"]
    result: Dict[str, Any] = {
        "session_id": sid,
        "profile": session_row.get("profile_name") or session_row.get("profile") or "?",
        "title": session_row.get("title"),
        "last_activity": session_row.get("last_activity"),
    }
    mem_dir = hermes_home / "context-memory" / re.sub(r"[^A-Za-z0-9_.-]", "_", sid)[:120]
    existing_state = mem_dir / "state.md"
    existing_transcript = mem_dir / "transcript.md"
    result["had_memory_files"] = existing_state.exists() or existing_transcript.exists()
    msgs = _load_active_messages(con, sid)
    result["message_count"] = len(msgs)
    result["message_count_db"] = session_row.get("message_count_db")
    if result["message_count_db"] is not None and result["message_count_db"] != len(msgs):
        result["message_count_note"] = (
            f"DB message_count={result['message_count_db']} vs active rows={len(msgs)}"
            " (inactive/compacted rows excluded)")
    turns, flags, notes = _group_turns(msgs)
    result["flags"] = flags
    result["notes"] = notes
    result["reconstructed_turns"] = len(turns)
    # Count turns already archived in existing files (idempotency reference).
    archived_count = 0
    if existing_transcript.exists():
        try:
            archived_count = len(re.findall(r"(?m)^## Turn ", existing_transcript.read_text(encoding="utf-8")))
        except Exception:
            archived_count = -1
            flags.append("existing transcript.md unreadable")
    result["archived_turns_existing"] = archived_count
    state_md, transcript_md, goal, n_turns = _build_files(sid, turns)
    result["first_user_message"] = goal[:120]
    result["first_user_verified"] = bool(goal) and not (
        "no user messages" in " ".join(flags))
    result["latest_archived_turn"] = n_turns
    result["state_size_bytes"] = len(state_md.encode())
    result["transcript_size_bytes"] = len(transcript_md.encode())
    if "no user messages" in " ".join(flags) and n_turns == 0:
        result["status"] = "empty"
        result["migration_status"] = "nothing to migrate (no active conversation rows)"
        return result
    if flags:
        result["status"] = "ambiguous"
        result["migration_status"] = "MANUAL REVIEW REQUIRED — not migrated"
        return result
    if archived_count and archived_count >= n_turns and existing_state.exists():
        result["status"] = "ready"
        result["migration_status"] = "already migrated (idempotent no-op)"
        return result
    if write:
        # Pre-validate BOTH targets before writing anything, so a blocked file
        # can never leave a half-migrated pair behind.
        if existing_state.exists() or existing_transcript.exists():
            result["status"] = "ambiguous"
            result["migration_status"] = (
                "MANUAL REVIEW REQUIRED — memory file(s) already exist, refusing to overwrite")
            return result
        try:
            mem_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(existing_transcript, transcript_md)
            _atomic_write(existing_state, state_md)
            result["migration_status"] = "written"
        except FileExistsError as exc:
            result["status"] = "ambiguous"
            result["migration_status"] = f"MANUAL REVIEW REQUIRED — {exc}"
            return result
        result["status"] = "ready"
    else:
        result["migration_status"] = "dry-run — no files written"
        result["status"] = "ready"
    return result


def main(argv: List[str]) -> int:
    args = list(argv)
    write = "--write" in args
    args = [a for a in args if a not in ("--write",)]
    do_reconcile = "--reconcile" in args
    args = [a for a in args if a != "--reconcile"]
    do_backfill = "--backfill" in args
    args = [a for a in args if a != "--backfill"]
    do_repair_state = "--repair-state" in args
    args = [a for a in args if a != "--repair-state"]
    sessions_filter: List[str] = []
    if "--session" in args:
        i = args.index("--session")
        sessions_filter = args[i + 1:]
        args = args[:i] + args[i + 1:]
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    con = _read_only_connection(hermes_home / "state.db")
    targets, stale = _target_sessions(con)
    if sessions_filter:
        targets = [t for t in targets if t["session_id"] in sessions_filter]
    report: List[Dict[str, Any]] = []
    for row in targets:
        if row.get("status") in ("missing", "archived"):
            row["migration_status"] = row.get("reason", "skipped")
            row["flags"] = row.get("flags", [])
            report.append(row)
            continue
        if do_reconcile:
            report.append(reconcile_session(row["session_id"], con, hermes_home,
                                            dry_run=not write))
        if do_backfill:
            report.append(backfill_session(row["session_id"], con, hermes_home,
                                           dry_run=not write))
        if do_repair_state:
            # Runs LAST: repair numbers the digest over CLOSED turns, which is
            # only aligned with the archive after reconcile+backfill complete.
            report.append(repair_state_session(row["session_id"], con, hermes_home,
                                               dry_run=not write))
            continue
        if do_reconcile or do_backfill:
            continue
        report.append(bootstrap_session(row, con, hermes_home, write))
    print("=" * 78)
    print(f"markdown-memory bootstrap — {'WRITE MODE' if write else 'DRY-RUN (no files written)'}")
    print(f"database: {hermes_home / 'state.db'} (opened READ-ONLY)")
    print("=" * 78)
    for r in report:
        print(f"\n[{r.get('status', '?').upper()}] {r['session_id']}  profile={r.get('profile')}")
        for k in ("title", "last_activity", "message_count", "message_count_db",
                  "message_count_note", "reconstructed_turns", "archived_turns_existing",
                  "first_user_message", "first_user_verified", "latest_archived_turn",
                  "state_size_bytes", "transcript_size_bytes", "had_memory_files",
                  "archived_turns", "db_turns", "reconciled_turns", "already_reconciled",
                  "live_turn_note", "backfilled_turns", "would_append", "closed_turns",
                  "repair_turns", "backup"):
            if r.get(k) is not None:
                print(f"  {k}: {r[k]}")
        for f in r.get("flags", []):
            print(f"  FLAG: {f}")
        for n in r.get("notes", []):
            print(f"  note: {n}")
        print(f"  migration: {r.get('migration_status')}")
    print("\n" + "-" * 78)
    print(f"STALE SESSIONS ({len(stale)}) — not migrated, listed for review:")
    for s in stale:
        print(f"  {s['session_id']}  src={s['source']} profile={s['profile']} "
              f"msgs={s['message_count']} in_tok={s['input_tokens']} "
              f"archived={s['archived']} last={s['last_activity']} title={s['title']!r}")
    ambiguous = [r for r in report if r.get("status") == "ambiguous"]
    print("\n" + "-" * 78)
    print(f"SUMMARY: {len(report)} target session(s); ambiguous/manual-review: "
          f"{len(ambiguous) or 'none'}; stale (not migrated): {len(stale)}")
    if ambiguous:
        print("SESSIONS REQUIRING MANUAL REVIEW:")
        for r in ambiguous:
            print(f"  {r['session_id']}: {'; '.join(r.get('flags', []))}")
    print("\nNo native session records were modified. DB connection was read-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
