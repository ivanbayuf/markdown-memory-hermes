"""Tests for bootstrap.py: read-only DB, dry-run default, atomic + idempotent
writes, tool/parallel preservation, turn-number continuity, ambiguity flags."""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

import os
REPO = Path(os.environ.get("HERMES_REPO", str(Path.home() / ".hermes" / "hermes-agent")))
if REPO.is_dir() and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_PLUGIN_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def boot(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "mm_bootstrap", _PLUGIN_DIR / "bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mm_bootstrap"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return mod, tmp_path


def _fixture_db(path: Path):
    """Minimal state.db fixture with the schema subset bootstrap reads."""
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, profile_name TEXT,
      message_count INTEGER, input_tokens INTEGER, archived INTEGER DEFAULT 0,
      title TEXT, last_activity_at REAL);
    CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
      role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
      timestamp REAL, active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0,
      _compressed_summary INTEGER DEFAULT 0);
    CREATE TABLE gateway_routing (scope TEXT DEFAULT '', session_key TEXT,
      entry_json TEXT, updated_at REAL);
    """)
    return con


def _ins(con, sid, rows):
    for r in rows:
        con.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls,"
            " tool_name, timestamp, compacted, _compressed_summary) VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, r.get("role"), r.get("content"), r.get("tool_call_id"),
             json.dumps(r["tool_calls"]) if r.get("tool_calls") else None,
             r.get("tool_name"), r.get("ts", 1700000000.0), r.get("compacted", 0),
             r.get("_compressed_summary", 0)))
    con.execute("INSERT INTO sessions (id, source, profile_name, message_count, input_tokens,"
                " archived, title, last_activity_at) VALUES (?,?,?,?,?,0,?,?)",
                (sid, "telegram", "default", len(rows), 123, "t", 1700000000.0))


def _route(con, sid, key="agent:main:telegram:dm:5401342146"):
    con.execute("INSERT OR REPLACE INTO gateway_routing (session_key, entry_json, updated_at)"
                " VALUES (?, ?, 1)", (key, json.dumps({"session_id": sid})))


def _turn_rows(seed="a"):
    """One completed turn with two sequential tool calls + a parallel pair turn."""
    def call(cid, name):
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}
    return [
        {"role": "user", "content": f"question one {seed}"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "read")]},
        {"role": "tool", "tool_call_id": "c1", "tool_name": "read", "content": "result one"},
        {"role": "assistant", "content": "", "tool_calls": [call("c2", "read")]},
        {"role": "tool", "tool_call_id": "c2", "tool_name": "read", "content": "result two"},
        {"role": "assistant", "content": f"final one {seed}"},
        {"role": "user", "content": f"question two {seed}"},
        {"role": "assistant", "content": "", "tool_calls": [call("p1", "read"), call("p2", "read")]},
        {"role": "tool", "tool_call_id": "p1", "tool_name": "read", "content": "parallel one"},
        {"role": "tool", "tool_call_id": "p2", "tool_name": "read", "content": "parallel two"},
        {"role": "assistant", "content": f"final two {seed}"},
    ]


class TestDryRunDefault:
    def test_dry_run_writes_nothing(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", _turn_rows())
        _route(con, "sess1")
        con.commit(); con.close()
        assert mod.main([]) == 0
        assert not (home / "context-memory").exists()

    def test_write_creates_files(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", _turn_rows())
        _route(con, "sess1")
        con.commit(); con.close()
        assert mod.main(["--write"]) == 0
        d = home / "context-memory" / "sess1"
        state = (d / "state.md").read_text(encoding="utf-8")
        transcript = (d / "transcript.md").read_text(encoding="utf-8")
        assert "## Goal" in state and "question one a" in state
        assert "## Turn 2" in transcript          # two turns
        assert "TOOL_CALL read args" in transcript
        assert "TOOL_RESULT[c1 (read)]: result one" in transcript
        # parallel pair preserved with original ids
        assert "TOOL_RESULT[p1 (read)]: parallel one" in transcript
        assert "TOOL_RESULT[p2 (read)]: parallel two" in transcript
        assert transcript.index("TOOL_RESULT[p1 (read)]") < transcript.index("TOOL_RESULT[p2 (read)]")

    def test_idempotent_write_never_overwrites(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", _turn_rows())
        _route(con, "sess1")
        con.commit(); con.close()
        mod.main(["--write"])
        d = home / "context-memory" / "sess1"
        before_t = (d / "transcript.md").read_text(encoding="utf-8")
        before_s = (d / "state.md").read_text(encoding="utf-8")
        mod.main(["--write"])   # second run: no-op, files untouched
        assert (d / "transcript.md").read_text(encoding="utf-8") == before_t
        assert (d / "state.md").read_text(encoding="utf-8") == before_s
        assert not list(d.glob("*.tmp-bootstrap"))

    def test_partial_file_never_written(self, boot, monkeypatch):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", _turn_rows())
        _route(con, "sess1")
        con.commit(); con.close()
        d = home / "context-memory" / "sess1"
        d.mkdir(parents=True)
        # Existing state.md blocks creation: transcript must not be left half-written.
        (d / "state.md").write_text("PRE-EXISTING", encoding="utf-8")
        assert mod.main(["--write"]) == 0
        assert (d / "state.md").read_text(encoding="utf-8") == "PRE-EXISTING"
        assert not (d / "transcript.md").exists()  # atomic: nothing partial
        assert not list(d.glob("*.tmp-bootstrap"))


class TestTurnNumbering:
    def test_engine_continues_after_bootstrap(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", _turn_rows())
        _route(con, "sess1")
        con.commit(); con.close()
        mod.main(["--write"])
        from conftest import load_engine_module
        engine_mod = load_engine_module()
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("sess1", hermes_home=str(home))
        assert e._turn_counter == 2
        assert "question one a" in e._digest
        # Next recorded turn is Turn 3 in the SAME archive file.
        e.on_turn_complete([
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ], turn_id="t3")
        transcript = e._archive_path().read_text(encoding="utf-8")
        assert "## Turn 3" in transcript
        assert "new question" in transcript


class TestAmbiguity:
    def test_orphaned_tool_result_flagged(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        rows = [{"role": "user", "content": "q"},
                {"role": "tool", "tool_call_id": "ghost", "tool_name": "read", "content": "x"},
                {"role": "assistant", "content": "a"}]
        _ins(con, "bad", rows)
        _route(con, "bad")
        con.commit(); con.close()
        assert mod.main([]) == 0  # dry-run exit fine
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main([])
        out = buf.getvalue()
        assert "MANUAL REVIEW REQUIRED" in out
        assert "orphaned tool result" in out
        assert not (home / "context-memory" / "bad").exists()

    def test_no_user_messages_flagged(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "orphan", [
            {"role": "assistant", "content": "unsolicited"},
            {"role": "user", "content": None} or {"role": "assistant", "content": "more"},
        ])
        _route(con, "orphan", key="agent:main:telegram:dm:OTHER")
        con.commit(); con.close()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main([])
        out = buf.getvalue()
        assert "MANUAL REVIEW REQUIRED" in out or "nothing to migrate" in out

    def test_first_row_not_user_flagged(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "lead", [
            {"role": "assistant", "content": "before any user"},
            {"role": "user", "content": "later user"},
            {"role": "assistant", "content": "a"},
        ])
        _route(con, "lead", key="agent:main:telegram:dm:OTHER2")
        con.commit(); con.close()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main([])
        out = buf.getvalue()
        assert "first active conversation row is role='assistant'" in out

    def test_compacted_history_preserved_and_noted(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        rows = _turn_rows("c")
        rows[0]["compacted"] = 1
        rows[2]["_compressed_summary"] = 1
        _ins(con, "comp", rows)
        _route(con, "comp", key="agent:main:telegram:dm:OTHER3")
        con.commit(); con.close()
        assert mod.main(["--write"]) == 0
        transcript = (home / "context-memory" / "comp" / "transcript.md").read_text(encoding="utf-8")
        assert "result one" in transcript  # compacted row content preserved as-is

    def test_stale_sessions_listed_not_migrated(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "routed", _turn_rows())
        _route(con, "routed")
        _ins(con, "stale_s", [{"role": "user", "content": "old"}])
        con.execute("UPDATE sessions SET last_activity_at=1000 WHERE id='stale_s'")
        con.commit(); con.close()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main(["--write"])
        out = buf.getvalue()
        assert "STALE SESSIONS" in out and "stale_s" in out
        assert not (home / "context-memory" / "stale_s").exists()

    def test_readonly_db_never_written(self, boot):
        """The DB must not gain a -journal write from bootstrap (mode=ro enforced)."""
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", _turn_rows())
        _route(con, "sess1")
        con.commit(); con.close()
        data_before = (home / "state.db").read_bytes()
        mod.main(["--write"])
        assert (home / "state.db").read_bytes() == data_before


class TestReconcile:
    """--reconcile: complete archived turns that were partial at snapshot time."""

    def _partial_fixture(self, boot):
        """Snapshot race: turn 1 was mid-flight when bootstrapped; its tool
        result + final were written to the DB only AFTER the snapshot."""
        mod, home = boot
        con = _fixture_db(home / "state.db")
        rows = [
            (1, "user", "q1", None),
            (2, "assistant", "", [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
            (5, "user", "live question", None),
            (6, "assistant", "", [{"id": "cl", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
        ]
        for rid, role, content, tcs in rows:
            con.execute(
                "INSERT INTO messages (id, session_id, role, content, tool_calls, timestamp)"
                " VALUES (?,?,?,?,?,?)",
                (rid, "sess1", role, content, json.dumps(tcs) if tcs else None, 1700000000.0))
        con.execute("INSERT INTO sessions (id, source, profile_name, message_count, input_tokens,"
                    " archived, title, last_activity_at) VALUES ('sess1','telegram','default',4,123,0,'t',1700000000.0)")
        _route(con, "sess1")
        con.commit(); con.close()
        assert mod.main(["--write"]) == 0  # snapshot: turn 1 archived partial
        d = home / "context-memory" / "sess1"
        tr = (d / "transcript.md").read_text(encoding="utf-8")
        assert "late-result" not in tr and "late-final" not in tr
        # The turn completes in the DB after the snapshot (rows 3, 4).
        con = sqlite3.connect(home / "state.db")
        con.execute("INSERT INTO messages (id, session_id, role, content, tool_call_id,"
                    " tool_name, timestamp) VALUES (3,'sess1','tool','late-result','c1','read',1700000001.0)")
        con.execute("INSERT INTO messages (id, session_id, role, content, tool_call_id,"
                    " tool_name, timestamp) VALUES (4,'sess1','assistant','late-final',NULL,NULL,1700000002.0)")
        con.commit(); con.close()
        return mod, home, d, tr

    def test_dry_run_reports_without_writing(self, boot):
        mod, home, d, _ = self._partial_fixture(boot)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main(["--reconcile", "--session", "sess1"])
        out = buf.getvalue()
        assert "reconciled_turns: [1]" in out
        assert "late-result" not in (d / "transcript.md").read_text(encoding="utf-8")

    def test_write_appends_completion_counter_unchanged(self, boot):
        mod, home, d, _ = self._partial_fixture(boot)
        mod.main(["--reconcile", "--write", "--session", "sess1"])
        tr = (d / "transcript.md").read_text(encoding="utf-8")
        assert "#### Turn 1 — completion" in tr
        assert "TOOL_RESULT[c1]: late-result" in tr
        assert "ASSISTANT: late-final" in tr
        assert "TOOL_RESULT[cl]" not in tr  # live turn untouched
        assert len(re.findall(r"(?m)^## Turn ", tr)) == 2  # counter headings unchanged
        from conftest import load_engine_module
        e = load_engine_module().MarkdownMemoryEngine()
        e.on_session_start("sess1", hermes_home=str(home))
        assert e._turn_counter == 2  # completion headings are NOT counted

    def test_idempotent(self, boot):
        mod, home, d, _ = self._partial_fixture(boot)
        mod.main(["--reconcile", "--write", "--session", "sess1"])
        before = (d / "transcript.md").read_text(encoding="utf-8")
        mod.main(["--reconcile", "--write", "--session", "sess1"])
        assert (d / "transcript.md").read_text(encoding="utf-8") == before

    def test_recall_finds_completion(self, boot):
        mod, home, d, _ = self._partial_fixture(boot)
        mod.main(["--reconcile", "--write", "--session", "sess1"])
        from conftest import load_engine_module
        e = load_engine_module().MarkdownMemoryEngine()
        e.on_session_start("sess1", hermes_home=str(home))
        out = json.loads(e.handle_tool_call("context_recall", {"query": "late-result"}))
        assert out["matches"] and "late-result" in out["matches"][0]

    def test_never_touches_production_without_flag(self, boot):
        """Reconcile is opt-in: plain bootstrap run must not modify archives."""
        mod, home, d, tr_before = self._partial_fixture(boot)
        mod.main(["--write"])
        assert (d / "transcript.md").read_text(encoding="utf-8") == tr_before


class TestBackfillAndRepair:
    def _seed_snapshot(self, boot):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        old = 1700000000.0
        _ins(con, "sess1", [
            {"role": "user", "content": "first goal", "ts": old},
            {"role": "assistant", "content": "first answer", "ts": old},
        ])
        _route(con, "sess1")
        con.commit(); con.close()
        assert mod.main(["--write"]) == 0
        con = sqlite3.connect(home / "state.db")
        con.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    ("sess1", "user", "backfilled question", old))
        con.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    ("sess1", "assistant", "backfilled answer", old))
        con.commit(); con.close()
        return mod, home

    def test_backfill_closed_turn_and_engine_counter(self, boot):
        mod, home = self._seed_snapshot(boot)
        assert mod.main(["--backfill", "--write", "--session", "sess1"]) == 0
        d = home / "context-memory" / "sess1"
        text = (d / "transcript.md").read_text(encoding="utf-8")
        assert "## Turn 2" in text and "USER: backfilled question" in text
        from conftest import load_engine_module
        e = load_engine_module().MarkdownMemoryEngine()
        e.on_session_start("sess1", hermes_home=str(home))
        assert e._turn_counter == 2

    def test_backfill_live_tool_turn_is_untouched(self, boot):
        mod, home = self._seed_snapshot(boot)
        con = sqlite3.connect(home / "state.db")
        con.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    ("sess1", "user", "live", 1700000000.0))
        con.execute("INSERT INTO messages (session_id, role, content, timestamp, tool_call_id) VALUES (?,?,?,?,?)",
                    ("sess1", "tool", "pending", 1700000000.0, "c-live"))
        con.commit(); con.close()
        mod.main(["--backfill", "--write", "--session", "sess1"])
        assert "USER: live" not in (home / "context-memory" / "sess1" / "transcript.md").read_text()

    def test_backfill_idempotent(self, boot):
        mod, home = self._seed_snapshot(boot)
        mod.main(["--backfill", "--write", "--session", "sess1"])
        path = home / "context-memory" / "sess1" / "transcript.md"
        before = path.read_bytes()
        mod.main(["--backfill", "--write", "--session", "sess1"])
        assert path.read_bytes() == before

    def test_repair_state_dry_run_and_write_preserves_goal(self, boot):
        mod, home = self._seed_snapshot(boot)
        state = home / "context-memory" / "sess1" / "state.md"
        original = state.read_bytes()
        transcript = (home / "context-memory" / "sess1" / "transcript.md").read_bytes()
        mod.main(["--repair-state", "--session", "sess1"])
        assert state.read_bytes() == original
        mod.main(["--repair-state", "--write", "--session", "sess1"])
        assert state.read_bytes() != original
        assert state.with_name("state.md.pre-repair.bak").read_bytes() == original
        assert "## Goal\nfirst goal\n" in state.read_text()
        assert (home / "context-memory" / "sess1" / "transcript.md").read_bytes() == transcript


class TestReviewRegressions:
    """Regression tests for the five defects found in the finishing review."""

    def _seed(self, boot, rows):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", rows)
        _route(con, "sess1")
        con.commit(); con.close()
        assert mod.main(["--write"]) == 0
        return mod, home, home / "context-memory" / "sess1"

    def _add(self, home, rows):
        con = sqlite3.connect(home / "state.db")
        for r in rows:
            con.execute(
                "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls,"
                " tool_name, timestamp) VALUES (?,?,?,?,?,?,?)",
                ("sess1", r["role"], r.get("content"), r.get("tool_call_id"),
                 json.dumps(r["tool_calls"]) if r.get("tool_calls") else None,
                 r.get("tool_name"), r.get("ts", 1700000000.0)))
        con.commit(); con.close()

    def test_reconcile_matches_blocks_by_user_text_not_position(self, boot):
        mod, home = boot
        call = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
        con = _fixture_db(home / "state.db")
        _ins(con, "sess1", [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "", "tool_calls": call},
            {"role": "tool", "tool_call_id": "c1", "tool_name": "read", "content": "late-result"},
            {"role": "assistant", "content": "late-final"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "live"},
        ])
        _route(con, "sess1")
        con.commit(); con.close()
        d = home / "context-memory" / "sess1"
        d.mkdir(parents=True)
        # Archive order differs from DB order: Turn 1 is q2 (complete),
        # Turn 2 is q1 (partial at snapshot time).
        (d / "transcript.md").write_text(
            "# Transcript archive — session sess1\n"
            "\n## Turn 1 — 2023-11-14 00:00:00 — status: bootstrapped\nUSER: q2\nASSISTANT: a2\n"
            "\n## Turn 2 — 2023-11-14 00:00:00 — status: bootstrapped\nUSER: q1\n"
            "TOOL_CALL read args: {}\n", encoding="utf-8")
        mod.main(["--reconcile", "--write", "--session", "sess1"])
        tr = (d / "transcript.md").read_text(encoding="utf-8")
        assert "#### Turn 2 — completion" in tr
        assert "#### Turn 1 — completion" not in tr
        completion = tr[tr.index("#### Turn 2 — completion"):]
        assert "TOOL_RESULT[c1]: late-result" in completion
        assert "ASSISTANT: late-final" in completion
        assert "USER: q2" not in completion

    def test_backfill_idempotent_with_empty_user_message(self, boot):
        mod, home, d = self._seed(boot, [
            {"role": "user", "content": "first goal"},
            {"role": "assistant", "content": "first answer"},
        ])
        self._add(home, [{"role": "user", "content": ""},
                         {"role": "assistant", "content": "answer to empty"}])
        mod.main(["--backfill", "--write", "--session", "sess1"])
        path = d / "transcript.md"
        before = path.read_bytes()
        assert before.count(b"ASSISTANT: answer to empty") == 1
        mod.main(["--backfill", "--write", "--session", "sess1"])
        assert path.read_bytes() == before

    def test_backfill_entries_match_engine_layout(self, boot):
        mod, home, d = self._seed(boot, [
            {"role": "user", "content": "first goal"},
            {"role": "assistant", "content": "first answer"},
        ])
        self._add(home, [{"role": "user", "content": "bq1"},
                         {"role": "assistant", "content": "ba1"},
                         {"role": "user", "content": "bq2"},
                         {"role": "assistant", "content": "ba2"}])
        mod.main(["--backfill", "--write", "--session", "sess1"])
        text = (d / "transcript.md").read_text(encoding="utf-8")
        # Engine writes "\n" + entry: every heading is preceded by a blank line.
        assert re.findall(r"(?m)^## Turn (\d+)", text) == ["1", "2", "3"]
        assert re.findall(r"\n\n## Turn (\d+)", text) == ["1", "2", "3"]
        assert text.endswith("ASSISTANT: ba2\n")

    def test_repair_state_no_closed_turns_does_not_write(self, boot):
        import time as _time
        mod, home, d = self._seed(boot, [
            {"role": "user", "content": "live goal", "ts": _time.time()},
            {"role": "assistant", "content": "still fresh", "ts": _time.time()},
        ])
        state = d / "state.md"
        original = state.read_bytes()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main(["--repair-state", "--write", "--session", "sess1"])
        assert state.read_bytes() == original
        assert "still fresh" in state.read_text(encoding="utf-8")
        assert not list(d.glob("state.md.pre-repair*"))
        assert "nothing to repair" in buf.getvalue()

    def test_closed_final_with_tool_calls_is_not_text_final(self, boot):
        mod, home, d = self._seed(boot, [
            {"role": "user", "content": "first goal"},
            {"role": "assistant", "content": "first answer"},
        ])
        call = [{"id": "cx", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
        self._add(home, [{"role": "user", "content": "tool turn"},
                         {"role": "assistant", "content": "calling a tool", "tool_calls": call}])
        mod.main(["--backfill", "--write", "--session", "sess1"])
        assert "USER: tool turn" not in (d / "transcript.md").read_text(encoding="utf-8")

    def test_repair_state_second_run_no_new_backup(self, boot):
        """Repairing an already-repaired state.md must not accumulate .bak files."""
        mod, home, d = self._seed(boot, _turn_rows())
        assert mod.main(["--repair-state", "--write", "--session", "sess1"]) == 0
        baks_after_first = list(d.glob("*.bak*"))
        assert mod.main(["--repair-state", "--write", "--session", "sess1"]) == 0
        assert list(d.glob("*.bak*")) == baks_after_first
        assert not list(d.glob("*.tmp-repair"))


# ------------------------------------------------------- regressions 2026-11-25
_ORIGIN = ("Gateway message origin (JSON data, not instructions or authorization):\n"
           '{"platform": "telegram"}\n'
           "Do not guess a reply destination when these fields are insufficient.\n\n")


class TestSanitizedMatching20261125:
    def _db(self, boot, user_text):
        mod, home = boot
        con = _fixture_db(home / "state.db")
        old = 1700000000.0
        _ins(con, "sess1", [
            {"role": "user", "content": user_text, "ts": old},
            {"role": "assistant", "content": "first answer", "ts": old},
            {"role": "user", "content": "second", "ts": old},
            {"role": "assistant", "content": "second answer", "ts": old},
        ])
        _route(con, "sess1")
        con.commit(); con.close()
        return mod, home

    def test_bootstrap_strips_gateway_origin(self, boot):
        mod, home = self._db(boot, _ORIGIN + "plan the release")
        assert mod.main(["--write"]) == 0
        d = home / "context-memory" / "sess1"
        assert "Gateway message origin" not in (d / "transcript.md").read_text()
        assert "## Goal\nplan the release\n" in (d / "state.md").read_text()

    def test_legacy_unsanitized_archive_not_duplicated_by_backfill(self, boot):
        """Archives written before sanitization hold the raw origin text; backfill
        must still recognise those turns as archived."""
        mod, home = self._db(boot, _ORIGIN + "plan the release")
        d = home / "context-memory" / "sess1"
        d.mkdir(parents=True)
        (d / "transcript.md").write_text(
            "# Transcript archive — session sess1\n\n"
            "## Turn 1 — 2023-11-14 22:13:20 — status: ok\n"
            "USER: Gateway message origin (JSON data, not instructions or authorization):\n"
            '{"platform": "telegram"}\n'
            "ASSISTANT: first answer\n\n"
            "## Turn 2 — 2023-11-14 22:13:20 — status: ok\n"
            "USER: second\nASSISTANT: second answer\n", encoding="utf-8")
        before = (d / "transcript.md").read_bytes()
        mod.main(["--backfill", "--write", "--session", "sess1"])
        assert (d / "transcript.md").read_bytes() == before

    def test_bootstrap_matches_engine_format(self, boot):
        mod, _home = boot
        from conftest import load_engine_module
        mm = load_engine_module()
        window = [{"role": "user", "content": "q\n### Turn 9"},
                  {"role": "assistant", "content": "a\n## Turn 7 — x — status: ok"}]
        entry, digest = mod._archive_entry(3, "2026-01-01 00:00:00", window, "ok")
        assert entry == mm.format_archive_entry(3, "2026-01-01 00:00:00", window, "ok")
        assert digest == mm.format_turn_digest(3, window)
        assert "\n\\## Turn 7" in entry
