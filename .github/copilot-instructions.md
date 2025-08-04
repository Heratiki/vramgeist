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

## Developer Workflows
- **Install dependencies:**
  - `uv pip install .` (standard)
  - `uv pip install --group test` (dev/test)
  - `uv sync` (sync env)
- **Run application:**
  - `uv run vramgeist [path]`
  - `uv run -m vramgeist`
  - `python -m vramgeist`
- **Interactive mode:**
  - Run with no args: launches file browser UI.
  - Key bindings: Up/Down (move), Enter (open/select), F2/Space (process), Backspace/Left (parent), Home (home dir), . (toggle hidden), q/ESC (exit).
- **Testing:**
  - `uv run pytest` or `uv run pytest tests/`
  - Tests in `tests/` mirror main modules (e.g., `test_calc.py` for `calc.py`).

## Patterns & Conventions
- **JSON output:** Use CLI flags for machine-readable results.
- **Environment variables:** Used for CI/headless automation; see `config.py`.
- **Error handling:** Exits with code 130 on cancel; prints user-friendly errors for OOM/model issues.
- **No global state:** All logic is modular, testable, and stateless.
- **Integration:** Designed for easy scripting and automation (output, env vars).

## Key Files & Directories
- `src/vramgeist/`: Main source code.
- `tests/`: Pytest-based tests for each module.
- `README.md`, `CLAUDE.md`: Usage, setup, and workflow examples.

## External Dependencies
- **uv:** For dependency management and running commands.
- **pytest:** For testing.

## Example Workflow
1. Install with `uv pip install .`
2. Run `uv run vramgeist` (interactive) or `uv run vramgeist model.gguf` (direct).
3. Use file browser to select model/folder.
4. View calculated context length and hardware info.

---
For unclear or missing conventions, check `README.md` and `CLAUDE.md` for up-to-date examples. If a workflow or pattern is ambiguous, ask for clarification or review recent test files for best practices.
