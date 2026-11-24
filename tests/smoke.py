import importlib.util, tempfile, json, os, sys
spec = importlib.util.spec_from_file_location(
    "markdown_memory", "/home/ubuntu/.hermes/plugins/markdown-memory/__init__.py",
    submodule_search_locations=["/home/ubuntu/.hermes/plugins/markdown-memory"])
m = importlib.util.module_from_spec(spec)
sys.modules["markdown_memory"] = m
spec.loader.exec_module(m)
e = m.MarkdownMemoryEngine()
home = tempfile.mkdtemp()
e.on_session_start("sess/TEST-1", hermes_home=home)
msgs = [
 {"role":"system","content":"sys"},
 {"role":"user","content":"What's in file a.txt and b.txt?"},
 {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{\"p\":\"a.txt\"}"}}]},
 {"role":"tool","tool_call_id":"c1","content":"alpha"},
 {"role":"assistant","content":"","tool_calls":[{"id":"c2","type":"function","function":{"name":"read","arguments":"{\"p\":\"b.txt\"}"}}]},
 {"role":"tool","tool_call_id":"c2","content":"beta"},
 {"role":"assistant","content":"Files contain alpha and beta."},
]
sel = e.select_context(msgs)
print("FIRST TURN selection:", None if sel is None else [x["role"] for x in sel])
e.on_turn_complete(msgs, usage={"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}, turn_id="t1")
msgs2 = msgs + [{"role":"user","content":"Now summarize them again please."}]
sel2 = e.select_context(msgs2)
print([(x["role"], (x.get("content") or "")[:60]) for x in sel2])
print("--- state.md ---")
print(open(os.path.join(home, "context-memory", "sess_TEST-1", "state.md")).read())
print("--- recall ---")
print(e.handle_tool_call("context_recall", {"query": "beta"})[:300])
