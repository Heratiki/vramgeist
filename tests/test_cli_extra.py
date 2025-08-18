import json
import tempfile
import os
from types import SimpleNamespace

import pytest

from src.vramgeist import cli


def test_cli_json_output_monkeypatched(monkeypatch, capsys):
    # Prepare a fake gguf file
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        f.write(b'x' * 1024)
        path = f.name

    # Monkeypatch gguf functions to return deterministic metadata
    monkeypatch.setattr('src.vramgeist.gguf.estimate_model_size_mb', lambda p: 1024.0)
    monkeypatch.setattr('src.vramgeist.gguf.read_gguf_metadata', lambda p: ({'llama.block_count': 32}, []))

    # Monkeypatch hardware detection in UI (ui imported hw functions at module import)
    monkeypatch.setattr('src.vramgeist.ui.get_gpu_memory', lambda *args, **kwargs: 8192)
    monkeypatch.setattr('src.vramgeist.ui.get_system_memory', lambda: (16384, 12288))

    # Run CLI main with --json and the fake path
    # main expects sys.argv-style input; call main directly with args list
    try:
        import sys
        monkeypatch.setattr(sys, 'argv', ['vramgeist', path, '--json'])
        # Call main which will read sys.argv
        cli.main()
        # When running with --json, main prints JSON to stdout
        captured = capsys.readouterr()
        out = captured.out.strip()
        assert out
        parsed = json.loads(out)
        assert isinstance(parsed, dict)
    finally:
        os.unlink(path)
