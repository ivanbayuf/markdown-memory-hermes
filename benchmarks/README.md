# Benchmarks

## Token benchmark (reproducible)

```sh
python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt
.venv/bin/python benchmarks/bench_hermes.py
```

Runs the deterministic 20-turn × 3-request fixture (seed `20261124`) through
the real engine, counts both arms with tiktoken `cl100k_base`, and writes
`benchmarks/results/hermes-2026-11-24.json` (per-request values included).
Re-running produces byte-identical JSON. See `docs/BENCHMARKS.md` for the
numbers, what they do and don't establish, and how to reproduce the README's
earlier 306k → 59k estimate as a real measurement.

## Tests

```sh
HERMES_REPO=/path/to/hermes-agent pytest tests/ -q
```

Unit + bootstrap tests run without a Hermes checkout (one contract test needs
the real `agent` module — see `docs/BENCHMARKS.md`). Integration tests need
the Hermes checkout and a mock provider.
