# Copilot Instructions for VRAMGEIST

## Project Architecture
- **Language & Structure:** Python 3.10+, organized as a CLI tool in `src/vramgeist/`.
- **Main Components:**
  - `cli.py`: Entry point, argument parsing, interactive UI logic.
  - `file_browser.py`: Terminal-based file/folder navigation.
  - `gguf.py`: GGUF model metadata parsing.
  - `hw.py`: Hardware (VRAM/RAM) detection.
  - `calc.py`: Context length calculation logic.
  - `config.py`: Handles config/environment variables.
  - `ui.py`: Terminal UI rendering and key handling.
- **Data Flow:**
  - User selects file/folder (via CLI or browser) → GGUF metadata parsed → Hardware info read → Context length calculated → Results printed (optionally as JSON).

```md
# Copilot instructions — VRAMGEIST (concise)

Purpose: help AI code agents be productive editing, testing, and extending this CLI tool.

Quick view
- Language: Python 3.10+; package root: `src/vramgeist/`.
- Entry points: `src/vramgeist/cli.py` and `src/vramgeist/__main__.py`.
- UI: `src/vramgeist/ui.py` + `src/vramgeist/tui/` (interactive browser).

Big picture
- Pipeline: input path(s) → GGUF parsing (`gguf.py`) → hardware detection (`hw.py`) → calculation engine (`calc.py`) → output via CLI/UI (`cli.py`, `ui.py`).
- Design goal: deterministic, scriptable CLI with an optional interactive TUI. Prefer pure functions in `calc.py` + small side-effecting orchestrators in `cli.py`.

Developer workflows (discoverable)
- Install (uses `uv`): `uv sync` then `uv pip install .` or `uv pip install --group test` for dev deps. `python -m vramgeist` also works.
- Run (interactive): `vramgeist` or `uv run vramgeist` (no args opens TUI in `tui/app.py`).
- Run (file(s)): `vramgeist <path|pattern>`; `--json` for machine output.
- Tests: `uv run pytest` or `uv run pytest tests/` (pytest-based tests live in `tests/`).

Project-specific patterns
- Environment-driven automation: `VRAMGEIST_BROWSE_AUTOPATH` and `VRAMGEIST_BROWSE_CANCEL` are read by the browse/ui logic for CI headless flows — see `cli.py` and `tui/file_browser.py`.
- Defensive parsing: `gguf.py` validates magic/version and falls back to defaults (default layer count, hidden size). When metadata missing, algorithms in `calc.py` use conservative defaults (see constants in `gguf.py`/`calc.py`).
- Hardware detection fallbacks: `hw.py` tries multiple OS-specific probes (nvidia-smi, dxdiag, psutil). Treat these as best-effort and rely on override flags (`--vram-mb`, `--ram-gb`) for deterministic test runs.
- Exit codes: interactive cancel uses 130; other errors generally surface exceptions — tests assert behavior in `tests/test_cli_no_args.py` and `tests/test_hw.py`.

Integration & dependencies
-- Key runtime deps (see `pyproject.toml` / `uv.lock`): `rich`, `textual` (TUI), `psutil`. `textual` is used exclusively for the interactive TUI; imports are lazy inside `src/vramgeist/tui/*` to avoid hard extras at import time.
- External integrations: GPU queries (nvidia-smi), platform tools (dxdiag on Windows). Stub or mock these in unit tests — tests already include hw fallbacks.

How to help (practical agent actions)
- When changing calculations, update `tests/test_calc.py` first to express intended behavior (happy path + 1 fallback case).
- For UI changes, prefer small, isolated edits in `tui/` and run the tests; TUI code is UI-only — keep calculation logic in `calc.py`.
- Use environment overrides to make runs deterministic in CI (e.g., `VRAMGEIST_BROWSE_AUTOPATH`, `--vram-mb`).

Files to consult (high-value)
- Core: `src/vramgeist/cli.py`, `src/vramgeist/gguf.py`, `src/vramgeist/hw.py`, `src/vramgeist/calc.py`, `src/vramgeist/ui.py`, `src/vramgeist/__main__.py`.
- TUI: `src/vramgeist/tui/app.py`, `src/vramgeist/tui/file_browser.py`, `src/vramgeist/tui/state.py`.
- Tests: `tests/test_calc.py`, `tests/test_cli_no_args.py`, `tests/test_gguf.py`, `tests/test_hw.py`.

If unsure
- Prefer reading `README.md` and `CLAUDE.md` for usage examples (already in repo). When behavior differs from docs, update docs and tests together.

Examples (copyable patterns)
- Run headless and print JSON (useful when editing output format):
  `VRAMGEIST_BROWSE_AUTOPATH="model.gguf" uv run vramgeist --json`
- Override hardware in tests/dev runs:
  `uv run vramgeist model.gguf --vram-mb 24000 --ram-gb 32`

Edit/PR checklist for agents
- Update/extend tests in `tests/` covering both normal and fallback behavior.
- Keep calculation changes confined to `calc.py`; UI/orchestration changes in `cli.py`/`ui.py`.
- Use env overrides in CI and add docs entry in `README.md` or `CLAUDE.md` for new flags.

Please review and tell me any missing project-specific rules or conventions to include.

```

## Model compatibility checklist (for agents)

When adding support or improving accuracy for new models (e.g., `gpt-oss` or other emerging formats), follow this minimal checklist:

1. Inspect model metadata and file format
  - Start with `src/vramgeist/gguf.py` to see how metadata keys are parsed. If the model exposes HF-style metadata, prefer using it.
2. Add conservative fallbacks
  - If key metadata (layer count, hidden size, dtype) is missing, choose safe defaults used elsewhere in the repo (see constants in `gguf.py`/`calc.py`).
3. Add calculation hooks in `calc.py`
  - Keep functions pure: inputs (model metadata dict, hardware numbers) → outputs (max_context, vram_usage_estimates, reasons).
4. Add tests
  - Add one happy-path test in `tests/test_gguf.py` and one fallback/edge-case in `tests/test_calc.py`.
5. Use env overrides and deterministic flags in tests
  - Prefer `--vram-mb` / `--ram-gb` overrides and `VRAMGEIST_BROWSE_AUTOPATH` when testing TUI flows.

Hugging Face metadata note
- If relevant, leverage Hugging Face model card metadata (packaged with a model or retrievable via HF APIs) to improve detection and defaults. Only create a dedicated "model template" when a model family has metadata or layout significantly different from existing tested models.

Policy for new model templates
- Only create new model templates when the user explicitly requests support for a model family with materially different metadata or storage layout. Otherwise, prefer defensive parsing + unit tests to handle new keys.

