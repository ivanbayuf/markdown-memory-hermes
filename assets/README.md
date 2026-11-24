# Assets

- `banner.svg` / `banner.png` (1600×560): repository banner. The SVG is the
  editable source and what the README embeds; the PNG is the shareable /
  social-preview version of the same design.
- `flow.svg` / `flow.png` (1500×560): request, recording, and recall flow.
  Same layout in both formats.
- `benchmark-chart.svg` / `benchmark-chart.png` (1500×660): per-request token chart from
  `benchmarks/results/hermes-2026-11-24.json` (native replay vs plugin,
  20 turns × 3 requests, tiktoken `cl100k_base`).

Palette: deep purple (`#1B1035` → `#2E184E`) with an amber accent
(`#F5A524`), matching the tagline "Keep the work. Bound the replay."
Diagrams use Liberation Sans / Liberation Mono. PNGs are rendered from the
same layout constants as the SVGs; regenerate all five files with
`python3 benchmarks/build_assets.py` (requires Pillow and the benchmark JSON
in `benchmarks/results/`). PNGs are not committed to git. The GitHub
connection used for pushes carries text files only, so re-run the script
after cloning to get the PNG versions locally.
