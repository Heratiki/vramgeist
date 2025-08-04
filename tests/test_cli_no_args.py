from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path



def _run_module_entrypoint(env: dict[str, str]) -> tuple[int, str, str]:
    # Run "python -m vramgeist.cli" so it executes main()
    cmd = [sys.executable, "-m", "vramgeist.cli"]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
        cwd=str(Path(__file__).resolve().parents[1]),  # project root
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_autopath_prints_and_exits(tmp_path: Path) -> None:
    # Create a temp directory to serve as a valid path
    p = tmp_path / "some_dir"
    p.mkdir()
    env = {"VRAMGEIST_BROWSE_AUTOPATH": str(p)}
    code, out, err = _run_module_entrypoint(env)
    assert code == 0
    # Should print exactly the absolute path with no trailing newline
    assert out == str(p.resolve())
    assert err == ""


def test_cli_cancel_exits_130() -> None:
    env = {"VRAMGEIST_BROWSE_CANCEL": "1"}
    code, out, err = _run_module_entrypoint(env)
    assert code == 130
    # No stdout on cancel
    assert out == ""
