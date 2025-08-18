import subprocess
import os
import tempfile
from types import SimpleNamespace
import builtins

import pytest

import src.vramgeist.hw as hw


class FakeCompletedProcess:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_detect_nvidia_gpu_monkeypatched(monkeypatch):
    # Simulate nvidia-smi output
    def fake_run(*args, **kwargs):
        return FakeCompletedProcess(stdout="8192\n4096\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Force detection via nvidia
    res = hw._detect_nvidia_gpu(timeout=1.0, policy="max")
    assert res == 8192

    res_first = hw._detect_nvidia_gpu(timeout=1.0, policy="first")
    assert res_first == 8192


def test_detect_amd_gpu_rocm_smi(monkeypatch):
    # Simulate rocm-smi CSV output where bytes value is included
    csv_line = "GPU[0],vram,Total (B),16106127360\n"

    def fake_run(*args, **kwargs):
        # Return successful csv for first call
        return FakeCompletedProcess(stdout=csv_line, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = hw._detect_amd_gpu(timeout=1.0, policy="first")
    # 16106127360 bytes == 15360 MB
    assert res == 15360


def test_detect_dxdiag_parsing_and_cleanup(monkeypatch, tmp_path):
    # Create a fake dxdiag output file
    content = """
    ------------------
    Display Devices
    ------------------
    Card name: Fake GPU
    Dedicated memory: 4096 MB
    Video memory: 4096 MB
    Approx. Total Memory: 8 GB
    """

    tmp_file = tmp_path / "dxdiag.txt"
    tmp_file.write_text(content)

    # Monkeypatch subprocess.run to create the file when called
    def fake_run(*args, **kwargs):
        # Emulate dxdiag writing the file: the invoked command includes the temp filename
        return FakeCompletedProcess(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Monkeypatch tempfile.gettempdir to return the tmp_path
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    # Ensure os.path.exists and os.path.getsize behave normally (they will with tmp_file present)
    # Write the file that dxdiag would have created
    # The _detect_dxdiag_gpu constructs a filename using pid; to keep it simple, monkeypatch os.getpid
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    fake_name = os.path.join(str(tmp_path), f"vramgeist_dxdiag_{12345}.txt")
    # Ensure file is large enough (>1000 bytes) to satisfy dxdiag wait logic
    with open(fake_name, "w", encoding="utf-8") as f:
        f.write(content * 100)

    # Now call the parser path
    res = hw._detect_dxdiag_gpu(timeout=1.0, policy="first")
    # Should detect 4096 as dedicated memory
    assert res == 4096


def test_get_system_memory_with_psutil(monkeypatch):
    # Provide fake psutil module via monkeypatch import mechanism
    class FakeVmem:
        total = 16 * 1024 * 1024 * 1024
        available = 12 * 1024 * 1024 * 1024

    fake_psutil = SimpleNamespace(virtual_memory=lambda: FakeVmem)

    monkeypatch.setattr(hw, "_get_psutil", lambda: fake_psutil)

    total, avail = hw.get_system_memory()
    assert total == 16384
    assert avail == 12288


def test_get_gpu_bandwidth_by_name_precedence():
    # Ensure lookup uses name when provided
    bw = hw.get_gpu_bandwidth_gbps(gpu_name="RTX 4090", measured=False)
    assert bw == 1008.0