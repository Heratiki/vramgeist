# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VRAMGeist is a Python command-line tool that calculates maximum safe context lengths for GGUF (GPT-Generated Unified Format) models based on available VRAM. It helps ML practitioners optimize their model configurations for their hardware.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Install the package
uv pip install .

# Install in development mode
uv pip install -e .

# Install with test dependencies
uv pip install --group test

# Run the application (multiple methods)
uv run vramgeist <path_to_gguf_file_or_folder>
uv run -m vramgeist
python -m vramgeist

# Run tests
uv run pytest
uv run pytest tests/
```

### Usage Examples
```bash
# Interactive file browser (when no args provided)
uv run vramgeist

# Analyze single GGUF file
uv run vramgeist model.gguf

# Analyze all GGUF files in directory
uv run vramgeist /path/to/models/

# Pattern matching
uv run vramgeist *.gguf

# JSON output for automation
uv run vramgeist model.gguf --json

# Override hardware detection
uv run vramgeist model.gguf --vram-mb 24000 --ram-gb 32

# Automation with environment variables
VRAMGEIST_BROWSE_AUTOPATH="model.gguf" uv run vramgeist
VRAMGEIST_BROWSE_CANCEL=1 uv run vramgeist
```

## Architecture

### Modular Package Structure
The application is organized as a proper Python package in `src/vramgeist/`:

```
src/vramgeist/
├── cli.py          # Command-line interface and argument parsing
├── config.py       # Configuration management and profiles  
├── hw.py           # Hardware detection (GPU VRAM, system RAM)
├── gguf.py         # GGUF file parser and metadata extraction
├── calc.py         # VRAM/RAM calculation algorithms
├── ui.py           # Rich-based terminal UI and analysis display
├── tui/            # Textual-based terminal UI components
├── __init__.py     # Package exports
└── __main__.py     # Module entry point
```

**Core Layers:**
- **Hardware Layer** (`hw.py`): Cross-platform GPU detection (NVIDIA, AMD, Intel, Apple Silicon, Windows dxdiag)
- **File Parser** (`gguf.py`): Binary GGUF file parsing with defensive error handling
- **Calculation Engine** (`calc.py`): VRAM/RAM usage algorithms and context length optimization
- **Configuration** (`config.py`): Profiles (default, conservative, aggressive) and parameter management
- **UI Layer** (`ui.py`): Rich-based terminal interface with live updates and tables
- **CLI Interface** (`cli.py`): Argument parsing, file discovery, and orchestration

### Core Algorithm
```
VRAM = Model_VRAM + Context_VRAM + Overhead
Model_VRAM = model_size_mb × gpu_layers_ratio  
Context_VRAM = context_length × hidden_size × 2 × bytes_per_element / (1024²)
Overhead = ~500MB for llama.cpp operations
```

### Processing Pipeline
1. Cross-platform hardware detection (GPU VRAM, system RAM)
2. Model file discovery and size calculation
3. GGUF metadata parsing (layer count, model parameters)
4. VRAM/RAM usage calculations for multiple GPU layer configurations
5. Optimization recommendations with configurable safety margins
6. Results display via Rich UI or JSON output

## Key Dependencies

- **uv**: Package manager (faster than pip)
- **rich>=14.1.0**: Terminal UI framework for live updates, tables, and styling
- **psutil>=7.0.0**: Cross-platform system resource detection (RAM, system info)
- **textual>=5.2.0**: Modern terminal user interface framework for interactive file browser
- **pytest>=7.0.0**: Testing framework (dev dependency)
- **Python 3.10+**: Required for modern pathlib and typing features

## Input Handling

The application supports multiple input methods:
- **Interactive Mode**: Terminal file browser when no arguments provided
- **Single GGUF files**: Direct file path analysis
- **Directory scanning**: Finds all `.gguf` files recursively
- **Glob pattern matching**: Shell-style pattern expansion
- **Multiple arguments**: Process multiple files/patterns in single command
- **Automation Support**: Environment variables for CI/headless operation

## UI Architecture

Uses Rich library's Live display for static updates:
- **Layout System**: Header + Model Info + Results panels
- **Progress Tracking**: Step-by-step analysis with status updates
- **Color Scheme**: Purple/blue theme throughout
- **Tables**: Formatted VRAM analysis results with recommendations
- **Cross-Platform**: Windows console compatibility (no problematic Unicode)

## Important Constants

- Default layer count: 32 (when metadata unavailable)
- Default VRAM: 8192 MB (when nvidia-smi unavailable)
- Safety margin: 90% of available VRAM
- Hidden size assumption: 4096 (typical for 7B-13B models)
- Bytes per element: 2 (fp16)
- llama.cpp overhead: 500MB

## File Format Support

Only supports GGUF files. The GGUF parser handles:
- Magic number validation (b'GGUF')
- Version checking
- Metadata key-value extraction
- Layer count detection from metadata keys like `llama.block_count`

## Error Handling Patterns

- **Hardware Detection Fallbacks**: Graceful degradation when GPU detection fails
- **Cross-Platform Compatibility**: Multiple GPU detection methods (nvidia-smi, AMD, Intel, dxdiag)
- **Metadata Parsing**: Default assumptions when GGUF metadata unavailable
- **File Validation**: Existence and format checking before processing
- **Path Resolution**: Cross-platform path normalization

## Test Structure

Tests are organized in `tests/` directory:
- `test_calc.py`: VRAM calculation function tests with known expected values
- `test_cli_no_args.py`: CLI behavior and environment variable tests
- `test_gguf.py`: GGUF file parsing tests with edge cases
- `test_hw.py`: Hardware detection fallback tests

Run tests with: `uv run pytest`