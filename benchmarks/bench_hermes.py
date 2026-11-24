"""Reproducible token benchmark for markdown-memory-hermes.

Scenario (fixed, deterministic): 20 turns x 3 requests per turn
(initial user message + 2 tool rounds), the same shape as the
20-turn x 3-call scenario quoted in the README.

Fixture (per turn, deterministic text from a seeded generator):
  user message      ~300 chars
  assistant reply   ~600 chars
  1 tool call + result, result ~1100 chars
  => ~2000 chars / ~500 cl100k tokens of *new* content per turn.

What is measured, at the exact capture point the engine controls:
  * native:  the outbound `messages` list Hermes would send (system +
    full transcript replay + current turn so far), for every request.
  * plugin:  the list returned by `MarkdownMemoryEngine.select_context`
    for the same request (system + frozen digest + current turn).

Counting: tiktoken `cl100k_base` (pinned in requirements) applied to the
exact captured payloads (role + content serialised as `role: content`
lines, which is what the harness counts for both arms identically).
`chars/4` is reported alongside as a rough estimate only, never as the
headline number. Provider-billed tokens and prompt-cache behaviour are
NOT measured here -- that needs a live provider run; see
docs/BENCHMARKS.md.

Run:  python3 benchmarks/bench_hermes.py [--turns 20] [--rounds 3]
        [--out benchmarks/results/hermes-2026-11-24.json]
Requires: pytest is NOT needed; tiktoken is (see benchmarks/requirements.txt).
Deterministic: same seed -> byte-identical fixture and results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
SEED = 20261124
WORDS = (
    "design review refactor deploy pipeline cache session transcript "
    "decision blocker milestone requirement discovery endpoint schema "
    "migration rollback timeout retry budget token context memory "
    "agent tool result prompt system user assistant analysis test "
    "coverage release note changelog stakeholder approval gate".split()
)


def _text(rng: random.Random, n_chars: int) -> str:
    words = []
    total = 0
    while total < n_chars:
        w = rng.choice(WORDS)
        words.append(w)
        total += len(w) + 1
    return " ".join(words)[:n_chars]


def build_turns(rng: random.Random, n_turns: int):
    """One turn = user msg, assistant msg w/ 1 tool call, tool result."""
    turns = []
    for t in range(1, n_turns + 1):
        user = f"Turn {t}: {_text(rng, 290)}"
        assistant = f"Working on turn {t}. {_text(rng, 580)}"
        tool_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{t:02d}",
                "type": "function",
                "function": {"name": "run_checks", "arguments": f'{{"turn": {t}}}'},
            }],
        }
        result = {
            "role": "tool",
            "tool_call_id": f"call_{t:02d}",
            "name": "run_checks",
            "content": f"Result {t}: {_text(rng, 1090)}",
        }
        turns.append([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
            tool_call,
            result,
            {"role": "assistant", "content": f"Done turn {t}. {_text(rng, 120)}"},
        ])
    return turns


def count_tokens(messages) -> int:
    text = "\n".join(f"{m.get('role', '?')}: {m.get('content') or ''}" for m in messages)
    return len(ENC.encode(text))


def count_chars(messages) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "markdown_memory_bench", REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["markdown_memory_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "hermes-2026-11-24.json")
    args = ap.parse_args()

    rng = random.Random(SEED)
    turns = build_turns(rng, args.turns)
    system = {"role": "system", "content": "You are Hermes, a coding agent. " + _text(random.Random(SEED + 1), 760)}
    fixture_text = json.dumps(turns, sort_keys=True)
    fixture_sha = hashlib.sha256(fixture_text.encode()).hexdigest()[:16]

    mod = load_engine()
    tmp = Path(tempfile.mkdtemp(prefix="mm-bench-"))
    engine = mod.MarkdownMemoryEngine()
    engine.on_session_start("bench", str(tmp))

    native_tokens, plugin_tokens = [], []
    native_chars, plugin_chars = [], []
    digest_hashes_per_turn = []
    state_chars_per_turn = []
    history: list = [system]

    for t, turn in enumerate(turns, start=1):
        # The 3 requests of a turn see progressively more of the turn.
        # Round 1: user msg only. Round 2: + assistant tool call.
        # Round 3: + tool result (full turn so far).
        partials = [turn[:1], turn[:3], turn[:4]]
        turn_digest_hashes = set()
        for r in range(args.rounds):
            request = history + partials[r]
            native_tokens.append(count_tokens(request))
            native_chars.append(count_chars(request))
            selected = engine.select_context(request)
            effective = selected if selected is not None else request
            plugin_tokens.append(count_tokens(effective))
            plugin_chars.append(count_chars(effective))
            if selected is not None and len(selected) > 1:
                turn_digest_hashes.add(hashlib.sha256(
                    str(selected[1].get("content", "")).encode()).hexdigest()[:12])
        digest_hashes_per_turn.append(sorted(turn_digest_hashes))
        # Turn completes: record it (updates state.md + frozen digest).
        engine.on_turn_complete(history + turn)
        history = history + turn
        state_chars_per_turn.append(len((tmp / "context-memory" / "bench" / "state.md").read_text()))

    n = len(native_tokens)
    total_native = sum(native_tokens)
    total_plugin = sum(plugin_tokens)
    result = {
        "engine": "markdown-memory-hermes",
        "engine_commit": "see docs/BENCHMARKS.md for the pinned commit",
        "date": "2026-11-24",
        "tokenizer": "tiktoken cl100k_base",
        "seed": SEED,
        "fixture_sha256_16": fixture_sha,
        "scenario": {"turns": args.turns, "requests_per_turn": args.rounds, "requests": n,
                     "system_prompt_chars": len(system["content"]),
                     "new_content_per_turn_chars": "~2000 (user ~300 + assistant ~600 + tool result ~1100)"},
        "native": {"total_tokens": total_native, "mean_per_request": round(statistics.mean(native_tokens), 1),
                   "median_per_request": statistics.median(native_tokens),
                   "first_request_tokens": native_tokens[0], "last_request_tokens": native_tokens[-1],
                   "total_chars": sum(native_chars)},
        "plugin": {"total_tokens": total_plugin, "mean_per_request": round(statistics.mean(plugin_tokens), 1),
                   "median_per_request": statistics.median(plugin_tokens),
                   "first_request_tokens": plugin_tokens[0], "last_request_tokens": plugin_tokens[-1],
                   "total_chars": sum(plugin_chars)},
        "reduction": {"total_tokens_pct": round(100 * (1 - total_plugin / total_native), 1),
                      "last_request_tokens_pct": round(100 * (1 - plugin_tokens[-1] / native_tokens[-1]), 1),
                      "tokens_saved": total_native - total_plugin},
        "digest": {"frozen_within_turn": all(len(h) <= 1 for h in digest_hashes_per_turn),
                   "turns_with_exactly_one_digest_hash": sum(1 for h in digest_hashes_per_turn if len(h) == 1),
                   "state_md_chars_first_turn": state_chars_per_turn[0],
                   "state_md_chars_last_turn": state_chars_per_turn[-1],
                   "state_md_chars_max": max(state_chars_per_turn)},
        "notes": [
            "Synthetic deterministic fixture, not a reproduction of the README's real-conversation 306k->59k claim.",
            "Counts are tiktoken cl100k_base on the captured messages lists, both arms identically.",
            "context_recall cost is excluded (on-demand, zero when unused).",
            "Provider-billed tokens / prompt-cache hits are not measured here.",
        ],
        "per_request": {"native_tokens": native_tokens, "plugin_tokens": plugin_tokens},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
