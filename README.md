# VRAMGEIST

Calculate the maximum safe context length for GGUF models based on available VRAM.

## Overview

**VRAMGEIST** is a command-line tool designed to help users, researchers, and developers determine the optimal context length for running GGUF-based large language models (LLMs) on their hardware. By analyzing your available VRAM and model parameters, VRAMGEIST ensures you avoid out-of-memory errors and maximize model performance.

### Key Features

- **Automatic VRAM detection:** Reads your system's available VRAM and RAM.
- **GGUF model analysis:** Parses GGUF files to extract model metadata.
- **Context length calculation:** Computes the largest safe context length for inference.
- **Interactive file/folder browser:** Lets you select files or directories in the terminal.
- **JSON output:** For integration with scripts and automation.
- **Environment variable bypass:** Supports CI and headless automation.

## Installation

VRAMGEIST requires Python 3.10 or newer.

Install dependencies with [uv](https://github.com/astral-sh/uv):

```sh
uv pip install .
```

Or, for development and testing:

```sh
uv pip install --group test
```

## Usage

### 1. Interactive Mode

Run without arguments to launch the interactive browser:

```sh
vramgeist
```

You'll see a terminal UI to navigate your filesystem. Use the following keys:

- **Up/Down:** Move selection
- **Enter:** Open folder or show file metadata
- **F2/Space:** Select current folder or file for processing
- **Backspace/Left:** Go to parent directory
- **Home:** Jump to home directory
- **. (dot):** Toggle hidden files
- **q/ESC:** Cancel (exit code 130)

**Example workflow:**

1. Launch `vramgeist`.
2. Browse to your GGUF model file or a folder containing models.
3. Press F2 or Space to select and process.
4. VRAMGEIST analyzes the file/folder and prints results.

### 2. Direct CLI Usage

You can also specify a file or folder directly:

```sh
vramgeist path/to/model.gguf
vramgeist path/to/models/
vramgeist *.gguf
```

**Example:**

```sh
vramgeist llama-7b.gguf
```

This will output the maximum safe context length and other metadata for the model.

### 3. JSON Output

For scripting or integration, use the `--json` flag:

```sh
vramgeist model.gguf --json
```

Outputs results in machine-readable JSON.

### 4. Custom Parameters

Override detected values with CLI flags:

```sh
vramgeist model.gguf --hidden-size 5120 --vram-safety 0.85
```

### 5. Automation & Testing

To bypass the interactive browser in CI or scripts, use environment variables:

- **VRAMGEIST_BROWSE_AUTOPATH:**  
  If set to a valid path, prints the absolute path and exits with code 0.
- **VRAMGEIST_BROWSE_CANCEL=1:**  
  Simulates cancel, exits with code 130 and no output.

**Example:**

```sh
VRAMGEIST_BROWSE_AUTOPATH="model.gguf" vramgeist
```

## Real World Examples

### Example 1: Model Deployment

A researcher wants to deploy a GGUF model on a GPU server and needs to know the largest context length that won't cause out-of-memory errors:

```sh
vramgeist /models/llama-13b.gguf
```
Output:
```
Max safe context length: 4096 tokens
Model: llama-13b.gguf
VRAM available: 24 GB
```

### Example 2: Batch Analysis

A developer wants to analyze all GGUF models in a directory and output results as JSON for further processing:

```sh
vramgeist /models/ --json
```

### Example 3: CI Testing

In a CI pipeline, the interactive browser is bypassed:

```sh
VRAMGEIST_BROWSE_AUTOPATH="/models/llama-7b.gguf" vramgeist
```

## Optional Dependency

The interactive browser uses `textual` (declared in `pyproject.toml` / `uv.lock`) for the TUI. Imports are lazy so the package can be used programmatically without requiring TUI extras unless `vramgeist` is run interactively.

If `textual` is not installed and you attempt to open the TUI, you'll get a clear error message with installation instructions:

```sh
uv pip install textual
```

## Intended Use Cases

- **Model deployment:** Ensure safe context lengths for LLM inference.
- **Benchmarking:** Compare VRAM requirements across models.
- **Automation:** Integrate with scripts, CI/CD, or model management tools.
- **Research:** Quickly analyze GGUF model metadata and hardware compatibility.

## Troubleshooting

- **Python version errors:**  
  Ensure you are using Python 3.10 or newer.
- **Dependency errors:**  
  Install missing dependencies with `uv pip install .`
- **Interactive browser issues:**  
  Make sure your terminal supports `textual` (modern terminal emulators) and Unicode. If you encounter rendering issues, try a different terminal or update your `textual`/`rich` versions.

## Contributing

Pull requests and issues are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.