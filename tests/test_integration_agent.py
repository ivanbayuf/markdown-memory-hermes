"""Integration tests: markdown-memory engine wired into the live AIAgent pipeline.

Uses a real AIAgent with an MCP tool server and a mocked chat client (the same
pattern as tests/agent/test_local_tool_batches.py) to exercise:

* tool-call continuity on the actual wire payloads across turns and tool rounds
* old-history replacement with the digest
* in-flight plugin failure -> host fail-open (run completes)
* manual /compress no-op behaviour
* state recording via the real on_turn_complete seam

No core Hermes files are modified; the engine instance is injected onto the agent
exactly as agent_init would when ``context.engine: markdown-memory`` is configured.
"""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from conftest import load_engine_module


@pytest.fixture
def mcp_server(tmp_path):
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    server = tmp_path / "server.py"
    server.write_text('''import json, pathlib, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    result = {}
    if req["method"] == "initialize":
        result = {"protocolVersion": req["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "mm", "version": "1"}}
    elif req["method"] == "tools/list":
        result = {"tools": [{"name": "read_" + n, "description": "Read " + n,
                  "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
                  "annotations": {"readOnlyHint": True}} for n in ("alpha", "beta")]}
    elif req["method"] == "tools/call":
        name = req["params"]["name"]
        result = {"content": [{"type": "text", "text": name + "-result-CONTENT"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
''', encoding="utf-8")
    names = sorted(register_mcp_servers({"mmfixture": {
        "command": sys.executable, "args": [str(server)], "timeout": 10}}))
    assert len(names) == 2
    try:
        yield names
    finally:
        shutdown_mcp_servers()


def _tool_call(cid, name, args=None):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {"key": "value"})))


def _response(calls=None, text=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text or ("" if calls else "done"), tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")], model="test/model",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110))


class RequestRecorder:
    """Wraps the mocked ``create`` so each request payload is captured."""

    def __init__(self, agent, side_effect):
        self.requests = []
        queue = list(side_effect)

        def create(**kwargs):
            self.requests.append(copy.deepcopy(kwargs.get("messages") or []))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        agent.client = MagicMock()
        agent.client.chat.completions.create = MagicMock(side_effect=create)

    def payloads(self):
        return self.requests


def _make_agent(engine, mcp_names, tmp_path):
    from run_agent import AIAgent

    with __import__("unittest").mock.patch("agent.process_bootstrap.OpenAI"), \
         __import__("unittest").mock.patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(
            model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
            enabled_toolsets=["mcp-mmfixture"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, max_iterations=4,
            session_id=f"mm-integration-{id(engine)}")
    agent._cached_system_prompt = "Use the available tools."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    # Wire the engine exactly as agent_init does for context.engine: markdown-memory
    agent.context_compressor = engine
    engine.update_model("test/model", 200000, provider="test", api_mode="chat")
    engine.on_session_start(agent.session_id or "mm-test", hermes_home=str(tmp_path))
    return agent


def _no_orphans(payload):
    declared = set()
    for m in payload:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared.update(c["id"] for c in m["tool_calls"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in declared, "orphan tool result on wire"


class TestAgentIntegration:
    def test_two_turns_trim_and_keep_tool_pairs(self, engine_mod, mcp_server, tmp_path):
        names = mcp_server
        # Turn 1: tool call then final. Turn 2: two sequential tool rounds then final.
        side_effect = [
            _response([_tool_call("r1", names[0])]),   # turn 1: call read_alpha
            _response(text="turn-one-final"),          # turn 1: final
            _response([_tool_call("r2", names[0])]),   # turn 2: call read_alpha
            _response([_tool_call("r3", names[1])]),   # turn 2: call read_beta
            _response(text="turn-two-final"),          # turn 2: final
        ]
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        rec = RequestRecorder(agent, side_effect)

        first_result = agent.run_conversation("Read alpha and tell me what it contains.")
        # The gateway accumulates history by passing prior turns back in; do the same.
        second_result = agent.run_conversation(
            "Now read beta too and compare with alpha.",
            conversation_history=copy.deepcopy(first_result["messages"]),
        )
        payloads = rec.payloads()
        assert len(payloads) == 5
        for p in payloads:
            _no_orphans(p)
            assert p[0]["role"] == "system"

        # Turn-1 first request: untouched history (only one user message).
        assert [m["role"] for m in payloads[0]] == ["system", "user"]

        # Turn-2 requests: replaced context = system + digest + current-turn chain.
        for p in payloads[2:]:
            assert p[1]["role"] == "user"
            assert p[1]["content"].startswith("[markdown-memory working state")
            assert "turn-one-final" in p[1]["content"]
            assert "Read alpha and tell me what it contains." in p[1]["content"]
            # Old turn-1 rows are NOT on the wire beyond the digest.
            assert not any(m.get("role") == "user" and m is not p[1] and "tell me what" in (m.get("content") or "") for m in p)
        # Final turn-2 request carries both tool rounds, paired.
        last = payloads[-1]
        ids = {c["id"] for m in last for c in (m.get("tool_calls") or [])}
        assert {"r2", "r3"} <= ids
        result_ids = {m["tool_call_id"] for m in last if m.get("role") == "tool"}
        assert ids <= result_ids
        # Determinism: the digest block is byte-identical across turn-2 iterations.
        digests = {p[1]["content"] for p in payloads[2:]}
        assert len(digests) == 1

    def test_parallel_tool_calls_on_wire(self, engine_mod, mcp_server, tmp_path):
        names = mcp_server
        side_effect = [
            _response(text="warmup"),                                   # turn 1
            _response([_tool_call("pa", names[0]), _tool_call("pb", names[1])]),  # parallel
            _response(text="both done"),                                # final
        ]
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        rec = RequestRecorder(agent, side_effect)
        first_result = agent.run_conversation("Say warmup.")
        agent.run_conversation(
            "Read alpha and beta together.",
            conversation_history=copy.deepcopy(first_result["messages"]),
        )
        last = rec.payloads()[-1]
        _no_orphans(last)
        parallel = next(m for m in last if len(m.get("tool_calls") or []) == 2)
        assert [c["id"] for c in parallel["tool_calls"]] == ["pa", "pb"]
        result_ids = [m["tool_call_id"] for m in last if m.get("role") == "tool"]
        assert "pa" in result_ids and "pb" in result_ids

    def test_plugin_failure_falls_back(self, engine_mod, mcp_server, tmp_path):
        """A crashing engine must not break the run (host fail-open)."""
        names = mcp_server
        side_effect = [
            _response([_tool_call("f1", names[0])]),
            _response(text="fallback-ok"),
        ]
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        # Session started fine, then the engine explodes mid-session.
        engine.on_session_start("x", hermes_home="/proc/definitely/not/writable")
        assert engine._session_dir is None  # passive
        rec = RequestRecorder(agent, side_effect)
        result = agent.run_conversation("Read alpha.")
        assert result["final_response"] == "fallback-ok"
        # Requests went through unmodified (no digest injection, no crash).
        assert all(not any("[markdown-memory working state" in (m.get("content") or "") for m in p)
                   for p in rec.payloads())

    def test_engine_select_crash_fail_open(self, engine_mod, mcp_server, tmp_path):
        """select_context raising mid-turn: host catches, request ships unchanged."""
        names = mcp_server
        side_effect = [_response([_tool_call("g1", names[0])]), _response(text="crash-safe")]
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        engine.on_session_start("s", hermes_home=str(tmp_path))
        engine.on_turn_complete([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "prior answer"},
        ], turn_id="seed")
        # Corrupt internal state so _select raises.
        def boom(msgs):
            raise RuntimeError("boom")
        engine._select = boom
        rec = RequestRecorder(agent, side_effect)
        result = agent.run_conversation("Read alpha after prior context.")
        assert result["final_response"] == "crash-safe"
        assert all(not any("working state" in (m.get("content") or "") for m in p)
                   for p in rec.payloads())

    def test_on_turn_complete_recorded_via_host_seam(self, engine_mod, mcp_server, tmp_path):
        """The real run_conversation finalization feeds on_turn_complete."""
        names = mcp_server
        side_effect = [_response(text="recorded-final")]
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        RequestRecorder(agent, [_response(text="recorded-final")])
        agent.run_conversation("The user asked about integration records.")
        state = engine._state_path().read_text(encoding="utf-8")
        assert "The user asked about integration records." in state
        assert "recorded-final" in state

    def test_retry_sends_identical_prefix(self, engine_mod, mcp_server, tmp_path):
        """A failed provider call retried within a turn must replay byte-identical
        prefix (system + frozen digest + turn user) on the retry attempt."""
        names = mcp_server
        first_request = {"captured": []}

        class R(RequestRecorder):
            def __init__(self, agent, side_effect):
                super().__init__(agent, side_effect)

        side_effect = [
            ConnectionError("boom attempt 1"),                       # retried
            _response([_tool_call("rr", names[0])]),
            _response(text="retried-final"),
        ]
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        # Seed a prior recorded turn so the digest path is active.
        seed_msgs = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        engine.on_turn_complete(copy.deepcopy(seed_msgs), turn_id="seed")
        rec = R(agent, side_effect)
        result = agent.run_conversation(
            "Read alpha again after retry.",
            conversation_history=copy.deepcopy(seed_msgs),
        )
        assert result["final_response"] == "retried-final"
        assert len(rec.requests) >= 3  # failed attempt + retry + tool round + (final may reuse)
        _no_orphans(rec.requests[-1])
        # The two attempts of the same call carry identical prefix.
        a, b = rec.requests[0], rec.requests[1]
        assert len(rec.requests) == 3
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), "retry attempt must replay identical bytes"
        assert b[0]["role"] == "system"
        assert b[1]["role"] == "user" and b[1]["content"].startswith("[markdown-memory working state")
        assert "Read alpha again after retry." in b[1]["content"]

    def test_manual_compress_noop(self, engine_mod, mcp_server, tmp_path):
        names = mcp_server
        engine = engine_mod.MarkdownMemoryEngine()
        agent = _make_agent(engine, names, tmp_path)
        RequestRecorder(agent, [_response(text="hi")])
        result = agent.run_conversation("hello")
        msgs = copy.deepcopy(result["messages"])
        out = agent.context_compressor.compress(msgs, force=True)
        assert out == msgs
        assert agent.context_compressor.should_compress(10 ** 9) is False
