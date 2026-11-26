"""Plain-python driver: exercises the same scenario as the integration tests,
capturing wire payloads. Bypasses pytest so we can see where a hard exit happens."""
import copy
import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("HERMES_REPO", os.path.expanduser("~/.hermes/hermes-agent")))
sys.path.insert(0, _HERE)

os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="mm-driver-")
print("HERMES_HOME:", os.environ["HERMES_HOME"], flush=True)

from conftest import load_engine_module
engine_mod = load_engine_module()

server_dir = tempfile.mkdtemp()
server = os.path.join(server_dir, "server.py")
open(server, "w").write('''import json, sys
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
                  "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}} for n in ("alpha", "beta")]}
    elif req["method"] == "tools/call":
        name = req["params"]["name"]
        result = {"content": [{"type": "text", "text": name + "-result-CONTENT"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
''')
from tools.mcp_tool_discovery import register_mcp_servers
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
names = sorted(register_mcp_servers({"mmfixture": {
    "command": sys.executable, "args": [server], "timeout": 10}}))
print("mcp tools:", names, flush=True)

def _tool_call(cid, name):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps({"key": "value"})))

def _response(calls=None, text=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text or ("" if calls else "done"), tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")], model="test/model",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110))

side_effect = [
    _response([_tool_call("r1", names[0])]),
    _response(text="turn-one-final"),
    _response([_tool_call("r2", names[0])]),
    _response([_tool_call("r3", names[1])]),
    _response(text="turn-two-final"),
]

engine = engine_mod.MarkdownMemoryEngine()
from run_agent import AIAgent
with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
    agent = AIAgent(model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
        enabled_toolsets=["mcp-mmfixture"], quiet_mode=True, skip_context_files=True,
        skip_memory=True, max_iterations=4, session_id="mm-driver-test")
print("agent constructed", flush=True)
agent.client = MagicMock(side_effect=side_effect)
agent._cached_system_prompt = "Use the available tools."
agent._use_prompt_caching = False
agent.compression_enabled = False
agent.save_trajectories = False
agent.context_compressor = engine
engine.update_model("test/model", 200000, provider="test", api_mode="chat")
engine.on_session_start(agent.session_id or "mm-test", hermes_home=os.environ["HERMES_HOME"])

requests = []
original_create = agent.client.chat.completions.create
def recorder(**kwargs):
    requests.append(copy.deepcopy(kwargs.get("messages") or []))
    return original_create(**kwargs)
agent.client.chat.completions.create = MagicMock(side_effect=recorder)

r1 = agent.run_conversation("Read alpha and tell me what it contains.")
print("turn1 final:", r1.get("final_response"), flush=True)
r2 = agent.run_conversation("Now read beta too and compare with alpha.")
print("turn2 final:", r2.get("final_response"), flush=True)

def no_orphans(p):
    declared = set()
    for m in p:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared.update(c["id"] for c in m["tool_calls"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in declared, "orphan tool result"

print("payloads:", len(requests), flush=True)
for p in requests:
    print([m["role"] for m in p], flush=True)
    no_orphans(p)
for p in requests[2:]:
    assert p[1]["role"] == "user" and p[1]["content"].startswith("[markdown-memory working state")
    assert "turn-one-final" in p[1]["content"]
last = requests[-1]
ids = {c["id"] for m in last for c in (m.get("tool_calls") or [])}
assert {"r2", "r3"} <= ids, ids
result_ids = {m["tool_call_id"] for m in last if m.get("role") == "tool"}
assert ids <= result_ids
digests = {p[1]["content"] for p in requests[2:]}
assert len(digests) == 1
print("ALL INTEGRATION CHECKS PASSED", flush=True)
print(json.dumps(engine._state_path().read_text(), indent=1)[:600], flush=True)
shutdown_mcp_servers()
