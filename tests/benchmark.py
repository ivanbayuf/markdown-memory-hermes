"""Benchmark: input-token cost and prompt-cache behaviour, engine vs native.

Runs the SAME synthetic multi-turn, multi-round tool conversation twice through
the real request-assembly pipeline (real AIAgent + mock provider):

  A) native behaviour (no context engine replacement)
  B) markdown-memory engine installed

For every wire payload we measure a rough token estimate of the messages that
would be sent (agent.model_metadata.estimate_messages_tokens_rough) and check
prefix stability across the iterations of each turn.

Purely offline: no API keys, no network.
"""
import copy
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/ubuntu/.hermes/hermes-agent")
sys.path.insert(0, "/home/ubuntu/.hermes/plugins/markdown-memory/tests")

os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="mm-bench-")

from conftest import load_engine_module
from agent.model_metadata import estimate_messages_tokens_rough

N_TURNS = 20
TOOL_ROUNDS = 2  # tool calls per turn
TOOL_RESULT_WORDS = 120   # tool result size in "words" (~tokens)
FINAL_WORDS = 60          # assistant final size
USER_WORDS = 40           # user ask size


def _w(n, seed):
    return " ".join(f"{seed}{i}" for i in range(n))


def _tool_call(cid, name):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps({"key": "v"})))


class Responder:
    """Scripted provider: each turn = tool rounds (each returns a big result) + final."""
    def __init__(self):
        self.turn = 0
        self.round = 0

    def __call__(self, **kwargs):
        if self.round < TOOL_ROUNDS:
            cid = f"t{self.turn}r{self.round}"
            self.round += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[_tool_call(cid, "read_thing")]),
                finish_reason="tool_calls")], model="bench/model",
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))
        self.round = 0
        self.turn += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=_w(FINAL_WORDS, f"final{self.turn}-"), tool_calls=None),
            finish_reason="stop")], model="bench/model",
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))


def make_agent(engine, hermes_home):
    from run_agent import AIAgent
    with patch("agent.process_bootstrap.OpenAI"), \
         patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(model="bench/model", api_key="k", base_url="http://127.0.0.1:9/v1",
                        enabled_toolsets=None, quiet_mode=True, skip_context_files=True,
                        skip_memory=True, max_iterations=12, session_id="bench")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are a benchmark agent."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    if engine is not None:
        agent.context_compressor = engine
        engine.update_model("bench/model", 256000, provider="bench", api_mode="chat")
        engine.on_session_start("bench", hermes_home=hermes_home)
    agent.client.chat.completions.create = MagicMock(side_effect=Responder())
    return agent


def run_all(engine):
    hermes_home = os.environ["HERMES_HOME"]
    agent = make_agent(engine, hermes_home)
    requests = []
    orig = agent.client.chat.completions.create
    recorder_resp = Responder()

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs.get("messages") or []))
        return recorder_resp(**kwargs)

    agent.client.chat.completions.create = MagicMock(side_effect=create)

    history = []
    for t in range(N_TURNS):
        result = agent.run_conversation(
            _w(USER_WORDS, f"ask{t}-"), conversation_history=copy.deepcopy(history) if history else None)
        history = copy.deepcopy(result["messages"])
    return requests


def metrics(requests):
    total = 0
    per_request = []
    stable_iters = 0
    total_iters = 0
    for req in requests:
        n = estimate_messages_tokens_rough(req)
        total += n
        per_request.append(n)
    # prefix stability: group requests by turn (each turn = TOOL_ROUNDS + 1 requests)
    per_turn = TOOL_ROUNDS + 1
    for i in range(0, len(requests), per_turn):
        group = requests[i:i + per_turn]
        if len(group) < 2:
            continue
        # a request's prefix = all messages before the last tool round additions:
        # compare sha of the FIRST message and second-message role/content.
        def prefix_hash(req):
            h = hashlib.sha256()
            h.update(json.dumps(req[0], sort_keys=True).encode())
            for m in req[1:]:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    break
                h.update(json.dumps(m, sort_keys=True, default=str).encode())
            return h.hexdigest()[:16]
        hashes = {prefix_hash(g) for g in group}
        stable_iters += 1 if len(hashes) == 1 else 0
        total_iters += 1
    return total, per_request, stable_iters, total_iters


def bench(label, engine):
    requests = run_all(engine)
    total, per, stable, iters = metrics(requests)
    first = per[0]
    last = per[-1]
    avg_last5 = sum(per[-5:]) / 5
    print(f"--- {label} ---")
    print(f"requests: {len(requests)} ({N_TURNS} turns x {TOOL_ROUNDS + 1} calls)")
    print(f"estimated input tokens TOTAL: {total:,}")
    print(f"first request: {first:,}  last request: {last:,}  avg(last 5): {avg_last5:,.0f}")
    print(f"per-turn iterations with byte-identical prefix: {stable}/{iters}")
    return total


native_total = bench("A) NATIVE (full history every call)", None)
SP_HOME = os.environ["HERMES_HOME"]
engine_mod = load_engine_module()
engine = engine_mod.MarkdownMemoryEngine()
engine_total = bench("B) markdown-memory engine", engine)

print()
print(f"TOTAL input tokens: native {native_total:,} vs markdown-memory {engine_total:,}")
print(f"reduction: {100 * (1 - engine_total / native_total):.1f}%")
