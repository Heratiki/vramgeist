# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VRAMGeist is a Python command-line tool that calculates maximum safe context lengths for GGUF (GPT-Generated Unified Format) models based on available VRAM. It helps ML practitioners optimize their model configurations for their hardware.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Run the application
uv run vramgeist <path_to_gguf_file_or_folder>

# Install in development mode
uv pip install -e .
```

### Usage Examples
```bash
# Analyze single GGUF file
uv run vramgeist model.gguf

# Analyze all GGUF files in directory
uv run vramgeist /path/to/models/

# Pattern matching
uv run vramgeist *.gguf
```

## Architecture

### Single-File Application Structure
The entire application is contained in `vramgeist.py` with a modular function design:

- **GPU Detection Layer**: Uses `nvidia-smi` to detect available VRAM with fallbacks
- **GGUF Parser**: Custom binary file parser that reads GGUF headers and metadata
- **VRAM Calculator**: Implements oobabooga's VRAM estimation formula
- **Rich UI Layer**: Live-updating terminal interface with progress indicators

### Core Algorithm
```
VRAM = Model_VRAM + Context_VRAM + Overhead
Model_VRAM = model_size_mb × gpu_layers_ratio  
Context_VRAM = context_length × hidden_size × 2 × bytes_per_element / (1024²)
Overhead = ~500MB for llama.cpp operations
```

### Processing Pipeline
1. GPU VRAM detection via `nvidia-smi`
2. Model file size calculation
3. GGUF metadata parsing (layer count extraction)
4. VRAM usage calculations for multiple GPU layer configurations
5. Optimization recommendations with safety margins

## Key Dependencies

- **uv**: Package manager (faster than pip)
- **rich**: Terminal UI framework for live updates, tables, and styling
- **Python 3.9+**: Required for modern pathlib and typing features

## Input Handling

The application supports multiple input methods:
- Single GGUF files
- Directory scanning (finds all `.gguf` files)  
- Glob pattern matching
- Multiple arguments in single command

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

- Graceful fallbacks for missing nvidia-smi
- Default assumptions when metadata unavailable  
- File existence validation before processing
- Path resolution for cross-platform compatibility