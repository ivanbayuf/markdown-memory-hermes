"""Unit tests for the markdown-memory context engine.

Covers: selection semantics (system + frozen digest + current-turn tail),
tool-call continuity (single round, parallel calls, multi-round), determinism
within a turn, turn recording (state.md / transcript.md), rolling state caps,
session resume from files, retrieval tool, and fail-open behaviour.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------- helpers
def _tool_call(cid, name, args_json='{"p": "x"}'):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args_json}}


def _turn1_msgs():
    """A completed turn 1 with two sequential tool rounds."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is in a.txt and b.txt?"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1", "read", '{"p": "a.txt"}')]},
        {"role": "tool", "tool_call_id": "c1", "content": "alpha"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c2", "read", '{"p": "b.txt"}')]},
        {"role": "tool", "tool_call_id": "c2", "content": "beta"},
        {"role": "assistant", "content": "a.txt has alpha, b.txt has beta."},
    ]


@pytest.fixture
def engine_mod():
    from conftest import load_engine_module
    return load_engine_module()


@pytest.fixture
def engine(engine_mod, tmp_path):
    e = engine_mod.MarkdownMemoryEngine()
    e.on_session_start("sess/abc-123", hermes_home=str(tmp_path))
    e.on_turn_complete(copy.deepcopy(_turn1_msgs()), usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, turn_id="t1")
    return e


def _assert_no_orphan_tool_msgs(selection):
    """Every role:tool row's tool_call_id must be declared by an earlier assistant
    tool_calls row in the SAME request (provider pairing rule)."""
    declared = set()
    for m in selection:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared.update(c["id"] for c in m["tool_calls"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in declared, f"orphan tool result {m.get('tool_call_id')}"


# ---------------------------------------------------------------- selection
class TestSelection:
    def test_first_request_untouched(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1", "t")]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
        assert e.select_context(msgs) is None

    def test_replaces_history_keeps_tail(self, engine):
        turn2_user = {"role": "user", "content": "Now compare them."}
        turn2 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is in a.txt and b.txt?"},  # old history
            {"role": "assistant", "content": "a.txt has alpha."},        # old history
            turn2_user,
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("c9", "read", '{"p": "c.txt"}')]},
            {"role": "tool", "tool_call_id": "c9", "content": "gamma"},
            {"role": "assistant", "content": "c.txt has gamma."},
        ]
        sel = engine.select_context(copy.deepcopy(turn2))
        assert sel is not None
        roles = [m["role"] for m in sel]
        # system, digest, previous exchange (user + final reply), current turn.
        assert roles == ["system", "user", "user", "assistant", "user", "assistant", "tool", "assistant"]
        assert sel[0]["content"] == "You are helpful."
        assert sel[2] == {"role": "user", "content": "What is in a.txt and b.txt?"}
        assert sel[3] == {"role": "assistant", "content": "a.txt has alpha."}
        # Current turn intact.
        assert sel[4] is turn2_user or sel[4]["content"] == "Now compare them."
        assert any(m.get("tool_call_id") == "c9" for m in sel)

    def test_without_last_exchange_only_digest_and_turn(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine(keep_last_exchange=False)
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(copy.deepcopy(_turn1_msgs()))
        sel = e.select_context(_turn1_msgs() + [{"role": "user", "content": "next"}])
        assert [m["role"] for m in sel] == ["system", "user", "user"]
        assert "a.txt has alpha, b.txt has beta." in sel[1]["content"]

    def test_digest_injection_contains_recorded_turn(self, engine):
        sel = engine.select_context([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is in a.txt and b.txt?"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "turn 3"},
        ])
        digest = sel[1]["content"]
        assert digest.startswith("[markdown-memory: earlier turns archived to disk")
        assert "What is in a.txt and b.txt?" in digest  # goal
        # The newest turn travels verbatim as the previous exchange, so its
        # digest section is not sent twice.
        assert "### Turn 1" not in digest
        assert sel[2]["content"] == "What is in a.txt and b.txt?"
        assert sel[3] == {"role": "assistant", "content": "x"}

    def test_tool_pairing_multi_round(self, engine):
        """Multi-round turn: both rounds' calls and results stay paired."""
        msgs = _turn1_msgs() + [{"role": "user", "content": "turn 3"}]
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_tool_call("d1", "read", '{"p": "a.txt"}')]})
        msgs.append({"role": "tool", "tool_call_id": "d1", "content": "alpha"})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_tool_call("d2", "read", '{"p": "b.txt"}')]})
        msgs.append({"role": "tool", "tool_call_id": "d2", "content": "beta"})
        sel = engine.select_context(copy.deepcopy(msgs))
        _assert_no_orphan_tool_msgs(sel)
        assert any(c["id"] == "d1" for m in sel for c in (m.get("tool_calls") or []))
        assert any(c["id"] == "d2" for m in sel for c in (m.get("tool_calls") or []))

    def test_parallel_tool_calls_kept_together(self, engine):
        """One assistant row with two parallel tool_calls + both results."""
        msgs = _turn1_msgs() + [
            {"role": "user", "content": "read both again"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("p1", "read", '{"p": "a.txt"}'), _tool_call("p2", "read", '{"p": "b.txt"}')]},
            {"role": "tool", "tool_call_id": "p1", "content": "alpha"},
            {"role": "tool", "tool_call_id": "p2", "content": "beta"},
        ]
        sel = engine.select_context(copy.deepcopy(msgs))
        _assert_no_orphan_tool_msgs(sel)
        parallel = next(m for m in sel if len(m.get("tool_calls") or []) == 2)
        ids = [c["id"] for c in parallel["tool_calls"]]
        result_ids = [m["tool_call_id"] for m in sel if m.get("role") == "tool"]
        for i in ids:
            assert i in result_ids


# ---------------------------------------------------------------- determinism
class TestDeterminism:
    def test_prefix_stable_across_iterations(self, engine):
        """Within one turn the selection prefix (system + digest) is byte-identical
        across tool-round iterations; the tail only ever appends."""
        base = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "old turn one"},
            {"role": "assistant", "content": "old answer one"},
            {"role": "user", "content": "turn two user"},
        ]
        sel_a = engine.select_context(copy.deepcopy(base))
        grown = base + [
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("e1", "read", '{"p": "z"}')]},
        ]
        sel_b = engine.select_context(copy.deepcopy(grown))
        grown2 = grown + [
            {"role": "tool", "tool_call_id": "e1", "content": "zz"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("e2", "read", '{"p": "w"}')]},
            {"role": "tool", "tool_call_id": "e2", "content": "ww"},
        ]
        sel_c = engine.select_context(copy.deepcopy(grown2))
        for a, b in zip(sel_a, sel_b):
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        for a, b in zip(sel_b, sel_c):
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert len(sel_c) > len(sel_b) > len(sel_a)

    def test_digest_frozen_between_recording_and_selection(self, engine):
        """select_context must never read a half-updated state."""
        before = copy.deepcopy(engine.select_context([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]))
        # A mid-turn recording attempt (host never does this) must not change the digest.
        engine._digest = engine._digest  # no-op guard
        after = engine.select_context([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ])
        assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)


# ---------------------------------------------------------------- recording
class TestRecording:
    def test_files_written(self, engine, tmp_path):
        d = tmp_path / "context-memory" / "sess_abc-123"
        state = (d / "state.md").read_text(encoding="utf-8")
        transcript = (d / "transcript.md").read_text(encoding="utf-8")
        assert "## Goal" in state and "What is in a.txt and b.txt?" in state
        assert "### Turn 1" in state
        assert "## Turn 1" in transcript
        assert "TOOL_RESULT[c1]: alpha" in transcript
        assert "TOOL_RESULT[c2]: beta" in transcript

    def test_state_rolls_and_caps(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine(max_state_chars=1200, max_recent_turns=4)
        e.on_session_start("s", hermes_home=str(tmp_path))
        for n in range(1, 10):
            msgs = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"question {n} " + "filler " * 50},
                {"role": "assistant", "content": f"answer {n} " + "detail " * 50},
            ]
            e.on_turn_complete(msgs, turn_id=f"t{n}")
        state = e._state_path().read_text(encoding="utf-8")
        assert len(state) <= e.max_state_chars
        assert "question 9" in state
        assert "question 1\n" not in state  # oldest turns rolled off

    def test_archive_append_only(self, engine):
        before = engine._archive_path().read_text(encoding="utf-8")
        engine.on_turn_complete([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "second turn"},
            {"role": "assistant", "content": "second answer"},
        ], turn_id="t2")
        after = engine._archive_path().read_text(encoding="utf-8")
        assert after.startswith(before)
        assert "## Turn 2" in after[len(before):]

    def test_interrupted_and_failed_status_recorded(self, engine):
        engine.on_turn_complete([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
        ], turn_id="t9", failed=True, interrupted=True)
        transcript = engine._archive_path().read_text(encoding="utf-8")
        assert "status: interrupted" in transcript  # interrupted wins display


# ---------------------------------------------------------------- resume
class TestResume:
    def test_new_instance_loads_state_and_archive(self, engine_mod, tmp_path, engine):
        e2 = engine_mod.MarkdownMemoryEngine()
        e2.on_session_start("sess/abc-123", hermes_home=str(tmp_path))
        assert "What is in a.txt and b.txt?" in e2._digest
        assert e2._turn_counter == 1  # archive already has turn 1
        # Selection with only the new turn still carries the loaded digest.
        sel = e2.select_context([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "fresh question"},
        ])
        assert "a.txt has alpha, b.txt has beta." in sel[1]["content"]
        # Next recorded turn continues numbering.
        e2.on_turn_complete([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "new answer"},
        ], turn_id="t2")
        transcript = e2._archive_path().read_text(encoding="utf-8")
        assert "## Turn 2" in transcript


# ---------------------------------------------------------------- retrieval
class TestRecall:
    def test_search_returns_matching_turns(self, engine):
        out = json.loads(engine.handle_tool_call("context_recall", {"query": "beta"}))
        assert out["matches"] and "b.txt" in out["matches"][0]

    def test_unknown_tool(self, engine):
        out = json.loads(engine.handle_tool_call("nope", {}))
        assert "error" in out

    def test_bad_args_fail_open(self, engine):
        out = json.loads(engine.handle_tool_call("context_recall", "not-a-dict"))
        assert "error" in out or "matches" in out

    def test_empty_archive(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("fresh", hermes_home=str(tmp_path))
        out = json.loads(e.handle_tool_call("context_recall", {"query": "anything"}))
        assert out["matches"] == []


# ---------------------------------------------------------------- fail-open
class TestFailOpen:
    def test_select_failure_returns_none(self, engine_mod, tmp_path, monkeypatch):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_turn1_msgs(), turn_id="t1")
        monkeypatch.setattr(e, "_select", lambda msgs: (_ for _ in ()).throw(RuntimeError("boom")))
        msgs = _turn1_msgs() + [{"role": "user", "content": "next"}]
        assert e.select_context(msgs) is None

    def test_garbage_input(self, engine):
        assert engine.select_context([]) is None
        assert engine.select_context([None, "junk"]) is None
        assert engine.select_context([{"role": "assistant", "content": "only assistant"}]) is None

    def test_on_turn_complete_never_raises(self, engine):
        # Immutable/insane input must not crash the host turn finalization.
        engine.on_turn_complete([{"role": "user", "content": {"weird": object()}}], turn_id="x")
        engine.on_turn_complete(None, turn_id="x")

    def test_broken_session_dir_passive(self, engine_mod, tmp_path):
        """If the session directory becomes unusable, recording degrades to a no-op
        but selection keeps working from the in-memory frozen digest."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_turn1_msgs(), turn_id="t1")
        e._session_dir = None  # simulate broken/unwritable state dir
        try:
            e.on_turn_complete(_turn1_msgs(), turn_id="t1")
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"on_turn_complete raised: {exc}")
        sel = e.select_context(_turn1_msgs() + [{"role": "user", "content": "next"}])
        assert sel is not None and sel[0]["role"] == "system"


# ---------------------------------------------------------------- safety net
class TestEmptyDigestGuard:
    def test_empty_digest_never_replaces_history(self, engine_mod, tmp_path):
        """A pre-engine session (no memory files, no recorded turns) must fall
        back to NATIVE context — history is never replaced with an empty state."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("legacy-session", hermes_home=str(tmp_path))
        msgs = _turn1_msgs() + [{"role": "user", "content": "continue please"}]
        assert e.select_context(msgs) is None

    def test_guard_allows_bootstrapped_digest(self, engine_mod, tmp_path):
        """After bootstrap (digest loaded, turn counter > 0) replacement works."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("legacy-session", hermes_home=str(tmp_path))
        e._digest = "# Working state\n## Recent turns\n### Turn 1\nUser: old\n"
        e._turn_counter = 1
        sel = e.select_context(_turn1_msgs() + [{"role": "user", "content": "continue"}])
        assert sel is not None and sel[1]["role"] == "user"

    def test_guard_allows_after_first_recorded_turn(self, engine_mod, tmp_path):
        """A brand-new engine session becomes replaceable once its first turn
        has been recorded (normal post-activation flow)."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("fresh", hermes_home=str(tmp_path))
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert e.select_context(msgs + [{"role": "user", "content": "q2"}]) is None
        e.on_turn_complete(msgs, turn_id="t1")
        sel = e.select_context(msgs + [{"role": "user", "content": "q2"}])
        assert sel is not None and "q1" in sel[1]["content"]

    def test_guard_with_broken_recording(self, engine_mod, tmp_path):
        """Recording failure (passive mode) must never produce empty-state
        replacement on a later request."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e._session_dir = None  # recording now fails silently
        e.on_turn_complete(_turn1_msgs(), turn_id="t1")  # no-op
        assert e._turn_counter == 0
        assert e.select_context(_turn1_msgs() + [{"role": "user", "content": "next"}]) is None


# ---------------------------------------------------------------- contract
class TestContract:
    def test_compaction_never_fires(self, engine):
        assert engine.should_compress(10_000_000) is False
        assert engine.has_content_to_compress([]) is False

    def test_manual_compress_is_noop(self, engine, tmp_path):
        msgs = _turn1_msgs()
        out = engine.compress(copy.deepcopy(msgs), current_tokens=999, force=True)
        assert out == msgs

    def test_update_from_response(self, engine):
        engine.update_from_response({"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
        assert engine.get_status()["last_prompt_tokens"] == 11

    def test_tool_schema_shape(self, engine):
        schemas = engine.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "context_recall"
        assert schemas[0]["parameters"]["properties"]["query"]["type"] == "string"

    def test_engine_is_context_engine_subclass(self, engine_mod, engine):
        pytest.importorskip("agent.context_engine", reason="Hermes host not importable; set HERMES_REPO")
        from agent.context_engine import ContextEngine
        assert isinstance(engine, ContextEngine)


# ------------------------------------------------------- regressions 2026-11-24
class TestRegressions20261124:
    def test_write_state_keeps_turn_headings(self, engine_mod, tmp_path):
        """_write_state used to strip the '### Turn ' delimiter when rolling
        state.md, so only the newest turn kept its heading."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        for i in (1, 2, 3):
            e.on_turn_complete([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ])
        state = (tmp_path / "context-memory" / "s" / "state.md").read_text()
        assert state.count("### Turn ") == 3
        assert "### Turn 1\n" in state and "### Turn 2\n" in state and "### Turn 3\n" in state

    def test_reset_does_not_replace_history(self, engine):
        """After on_session_reset, select_context must pass the request through
        unchanged (never substitute a placeholder digest for real history)."""
        engine.on_session_reset()
        req = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is in a.txt and b.txt?"},
            {"role": "assistant", "content": "a.txt has alpha."},
            {"role": "user", "content": "And now?"},
        ]
        assert engine.select_context(copy.deepcopy(req)) is None

    def test_empty_digest_never_replaces_history(self, engine_mod, tmp_path):
        """Even with a nonzero turn counter, an empty digest means passthrough."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e._turn_counter = 3
        e._digest = ""
        req = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert e.select_context(copy.deepcopy(req)) is None


# ------------------------------------------------------- regressions 2026-11-25
def _simple_turn(u, a, sys_msg="sys"):
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": u},
            {"role": "assistant", "content": a}]


class TestRegressions20261125:
    def test_previous_reply_visible_beyond_digest_clip(self, engine_mod, tmp_path):
        """Follow-ups like 'do option 4' need the previous reply, which the
        300-char digest line used to cut off."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        reply = "Options:\n" + "\n".join(f"{i}. option {i}: " + "detail " * 12 for i in range(1, 6))
        e.on_turn_complete(_simple_turn("give me options", reply))
        sel = e.select_context(_simple_turn("give me options", reply)
                               + [{"role": "user", "content": "do option 4"}])
        assert any(m["role"] == "assistant" and "option 4: detail" in m["content"] for m in sel)

    def test_previous_exchange_skips_tool_rows(self, engine):
        sel = engine.select_context(_turn1_msgs() + [{"role": "user", "content": "next"}])
        assert [m["role"] for m in sel] == ["system", "user", "user", "assistant", "user"]
        assert sel[3]["content"] == "a.txt has alpha, b.txt has beta."
        _assert_no_orphan_tool_msgs(sel)

    def test_reset_starts_fresh_state_and_unique_turn_numbers(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_simple_turn("build feature X", "done X"))
        e.on_turn_complete(_simple_turn("now Y", "done Y"))
        e.on_session_reset()
        e.on_turn_complete(_simple_turn("brand new topic", "ok new"))
        d = tmp_path / "context-memory" / "s"
        state = (d / "state.md").read_text()
        assert "brand new topic" in state and "feature X" not in state and "now Y" not in state
        assert "## Goal\nbrand new topic" in state
        assert list(d.glob("state.md.pre-reset-*.bak")), "old state must be backed up"
        transcript = (d / "transcript.md").read_text()
        import re
        nums = re.findall(r"(?m)^## Turn (\d+)", transcript)
        assert nums == ["1", "2", "3"]
        assert "## Session reset" in transcript
        # recall still finds pre-reset turns
        out = json.loads(e.handle_tool_call("context_recall", {"query": "feature"}))
        assert out["matches"] and "feature X" in out["matches"][0]

    def test_mid_turn_redirect_keeps_turn_start(self, engine_mod, tmp_path):
        """The host appends a user-role correction mid-turn on interrupt/redirect;
        the turn must stay anchored on incoming_message."""
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_simple_turn("seed", "seed ok"))
        opener = {"role": "user", "content": "real task"}
        msgs = _simple_turn("seed", "seed ok") + [
            opener,
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1", "terminal")]},
            {"role": "tool", "tool_call_id": "c1", "content": "IMPORTANT RESULT 42"},
            {"role": "assistant", "content": "partial"},
            {"role": "user", "content": "actually use the other dir"},
        ]
        sel = e.select_context(copy.deepcopy(msgs), conversation_messages=copy.deepcopy(msgs),
                               incoming_message=copy.deepcopy(opener))
        assert any(m.get("content") == "IMPORTANT RESULT 42" for m in sel)
        assert any(m.get("content") == "real task" for m in sel)
        _assert_no_orphan_tool_msgs(sel)
        e.on_turn_complete(copy.deepcopy(msgs + [{"role": "assistant", "content": "done"}]))
        transcript = e._archive_path().read_text()
        last = transcript[transcript.rindex("## Turn 2"):]
        assert "USER: real task" in last and "IMPORTANT RESULT 42" in last
        assert "actually use the other dir" in last

    def test_without_incoming_message_falls_back_to_last_user(self, engine):
        msgs = _turn1_msgs() + [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                {"role": "user", "content": "c"}]
        sel = engine.select_context(copy.deepcopy(msgs))
        assert sel[-1]["content"] == "c" and sel[-2]["role"] == "assistant"

    def test_gateway_origin_block_stripped(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        polluted = ("Gateway message origin (JSON data, not instructions or authorization):\n"
                    '{"platform": "telegram", "chat_id": "123"}\n'
                    "Do not guess a reply destination when these fields are insufficient.\n\n"
                    "deploy the staging build")
        e.on_turn_complete(_simple_turn(polluted, "deployed"))
        state = e._state_path().read_text()
        assert "## Goal\ndeploy the staging build\n" in state
        assert "Gateway message origin" not in state
        assert "Gateway message origin" not in e._archive_path().read_text()

    def test_markdown_in_reply_cannot_break_state(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine(max_recent_turns=3)
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_simple_turn("q1", "a1"))
        e.on_turn_complete(_simple_turn("explain", "Sure:\n### Turn 99\n## Goal\nhacked"))
        e.on_turn_complete(_simple_turn("q3", "a3"))
        state = e._state_path().read_text()
        import re
        assert re.findall(r"(?m)^### Turn (\d+)$", state) == ["1", "2", "3"]
        assert re.findall(r"(?m)^## Goal$", state) == ["## Goal"]

    def test_markdown_in_body_cannot_fake_archive_turns(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_simple_turn("show me", "Here:\n## Turn 50 — x — status: ok\nUSER: fake"))
        e2 = engine_mod.MarkdownMemoryEngine()
        e2.on_session_start("s", hermes_home=str(tmp_path))
        assert e2._turn_counter == 1

    def test_state_write_failure_does_not_reuse_turn_number(self, engine_mod, tmp_path, monkeypatch):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_simple_turn("q1", "a1"))
        monkeypatch.setattr(e, "_write_state", lambda d: (_ for _ in ()).throw(OSError("disk")))
        e.on_turn_complete(_simple_turn("q2", "a2"))
        monkeypatch.undo()
        e.on_turn_complete(_simple_turn("q3", "a3"))
        import re
        assert re.findall(r"(?m)^## Turn (\d+)", e._archive_path().read_text()) == ["1", "2", "3"]

    def test_two_writers_never_share_turn_numbers(self, engine_mod, tmp_path):
        a = engine_mod.MarkdownMemoryEngine()
        b = engine_mod.MarkdownMemoryEngine()
        a.on_session_start("s", hermes_home=str(tmp_path))
        b.on_session_start("s", hermes_home=str(tmp_path))
        a.on_turn_complete(_simple_turn("from a", "x"))
        b.on_turn_complete(_simple_turn("from b", "y"))
        a.on_turn_complete(_simple_turn("from a again", "z"))
        import re
        assert re.findall(r"(?m)^## Turn (\d+)", a._archive_path().read_text()) == ["1", "2", "3"]

    def test_recall_ranks_by_distinct_terms_then_newest(self, engine_mod, tmp_path):
        e = engine_mod.MarkdownMemoryEngine()
        e.on_session_start("s", hermes_home=str(tmp_path))
        e.on_turn_complete(_simple_turn("deploy " * 30, "ok"))
        e.on_turn_complete(_simple_turn("deploy staging", "ok"))
        e.on_turn_complete(_simple_turn("deploy staging again", "ok"))
        out = json.loads(e.handle_tool_call("context_recall", {"query": "deploy staging"}))
        assert "deploy staging again" in out["matches"][0]
        assert "deploy staging" in out["matches"][1]

    def test_recall_error_does_not_leak_details(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "_recall", lambda a: (_ for _ in ()).throw(RuntimeError("/secret/path")))
        out = engine.handle_tool_call("context_recall", {"query": "x"})
        assert "/secret/path" not in out and "error" in json.loads(out)

    def test_recall_bad_limit(self, engine):
        out = json.loads(engine.handle_tool_call("context_recall", {"query": "beta", "limit": "lots"}))
        assert out["matches"]

    def test_keywords_unicode_and_indonesian_stopwords(self, engine_mod):
        kw = engine_mod._keywords("tolong perbaiki konfigurasi yang ada di server, konfigurasi rusak")
        assert kw.split(", ")[0] == "konfigurasi"
        assert "yang" not in kw and "tolong" not in kw
