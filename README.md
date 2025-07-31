# VRAMGEIST

Calculate maximum safe context length for GGUF models based on available VRAM.

## Python version

This project now requires Python 3.10 or newer. If you see a type hint error like "unsupported operand type(s) for |: 'type' and 'NoneType'", ensure your environment is using Python ≥3.10.

With uv:
- Install/select Python 3.10+ and re-run: `uv run vramgeist`

## Interactive file/folder browser (no-args mode)

When you run the CLI without any arguments:

    vramgeist

an interactive file/folder browser opens in the terminal to let you navigate and select a path.

Navigation keys:
- Up/Down: move selection
- Enter: open/select (descend or confirm if selectable)
- Backspace or Left: go to parent directory
- Home: jump to your home directory
- . (dot): toggle visibility of hidden files
- q or ESC: cancel (exit code 130)

When a selectable entry is chosen (files and directories are both selectable), the CLI prints the absolute path to stdout and exits with code 0.

If you provide arguments (paths, flags), existing behavior is preserved and the interactive browser is not invoked unless you specify no paths after options.

## Non-interactive testing/automation

To support CI and scripting without launching the UI, you can use these environment variables:

- VRAMGEIST_BROWSE_AUTOPATH: If set to a valid path, the CLI will print its absolute path to stdout and exit with code 0 when invoked without arguments.
- VRAMGEIST_BROWSE_CANCEL="1": Simulates cancel, causing the CLI to exit with code 130 and no output.

These variables enable deterministic behavior during tests and automation.

## Optional dependency

The interactive browser requires prompt_toolkit. It is declared as a dependency in pyproject.toml:

- prompt_toolkit>=3.0,<4.0

If prompt_toolkit is not available and you attempt to invoke the interactive mode, a clear error message will be shown indicating how to install it.