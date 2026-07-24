# AGENTS.md — Negative Color Corrector

Compact guidance for AI agents working in this repo.

## What this is

Python CLI/Web app that removes C-41 orange mask from film negative scans and outputs natural-color corrections. Core pipeline: invert → mask analysis → channel compensation → auto levels → warmth → AI cast detection (feedback loop).

## Quick commands

```bash
# Install deps
pip install numpy Pillow requests

# CLI correction (negative, no warmth)
python main.py scan.tiff -w none

# CLI correction (negative, natural warmth)
python main.py scan.tiff -w natural

# Batch mode
python main.py img1.tiff img2.tiff -w none

# Web UI
pip install gradio
python main.py --gui

# With VLM cast detection
python main.py scan.tiff -w none -d vlm_api --model "openai/gpt-4o-mini"
```

No test suite exists. No linter/formatter configured. No build step — pure Python.

## Architecture

```
main.py          — entry point (CLI argparse + GUI launch)
config.py        — default constants (DETECTOR_MODE, warmth params, thresholds)
core/            — all image processing logic
  engine.py      — Engine class: orchestrates the full pipeline, holds feedback loop
  invert.py      — pixel = 255 - pixel (negative mode only)
  mask_analyzer.py — 3-strategy reference point sampling (film base → neutral gray → gray-world)
  channel_comp.py — per-channel scale multiplication
  auto_levels.py — percentile-based histogram stretch
  warmth.py      — warmth presets (none/natural/kodak_gold/fuji_superia/cool)
  cast_detector.py — heuristic + VLM API backends, DetectorFactory
  model_provider.py — multi-provider VLM config (OpenRouter, etc.)
  credential_store.py — API key storage (see below)
  config_manager.py — JSON5 config (see below)
ui/
  app.py         — Gradio web interface
```

## Key conventions

- **Warmth direction**: always go yellow (R↑ + G↑), never magenta (R↑ + B↑). Magenta produces unnatural purple cast.
- **Channel precision**: work in float32. A 0.01 scale difference is visually significant.
- **Mask reference priority**: film base edge (left 5-10 cols) > in-scene neutral gray (algorithm, VLM-corrected when key available) > VLM direct定位 (algorithm fails + key available) > global gray-world fallback.
- **AI feedback loop**: max 3% adjustment per iteration, 10-iteration cap to prevent oscillation.
- **Config precedence**: `~/.negative-corrector/config.json5` (on this machine: `/mnt/c/Users/Lee_B/.negative-corrector/config.json5`) is base; CLI args override it.
- **Batch limit**: 40 images max per run.
- **Credentials**: stored in `.credentials.json` / `.credentials.enc` (gitignored).
- **Image I/O**: Pillow for loading/saving, NumPy arrays (float32) for processing. Output saved to `_corrected/` subfolder in the input directory.

## Gotchas

- `config.py` at repo root is legacy defaults; `core/config_manager.py` + `~/.negative-corrector/config.json5` (on this machine: `/mnt/c/Users/Lee_B/.negative-corrector/config.json5`) is the real config system used by the engine.
- No `__init__.py` tests or pytest setup — if you add tests, create them from scratch.
- `.tiff` and `.TIF` files are gitignored (test images / scan outputs).
- The `--no-cli` flag launches GUI (confusing name — it means "not CLI mode").
