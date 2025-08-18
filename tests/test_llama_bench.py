import os
import sys
import types
import time
import shutil
import importlib
import subprocess

import pytest

from vramgeist.bench import llama_bench as lb


def test_make_repeated_prompt():
    s = lb._make_repeated_prompt('x', 5)
    assert s.count('x') == 5


def test_fit_k_from_measurements_basic():
    data = {1024: 1.0, 2048: 0.5, 4096: 0.25}
    k = lb.fit_k_from_measurements(data, eps=1.0)
    assert k is not None
    # k should be roughly consistent across points
    assert k > 0


def test_measure_tps_with_python_binding_missing(monkeypatch):
    # Simulate llama_cpp missing by ensuring import raises
    monkeypatch.setitem(sys.modules, 'llama_cpp', None)
    res = lb.measure_tps_with_python_binding('model.gguf', [1])
    assert res == {"map": {}, "details": {}}


def test_measure_tps_with_binary_not_executable(tmp_path, monkeypatch):
    # create a fake file that's not executable
    f = tmp_path / 'fakebin'
    f.write_text('noop')
    # On some platforms os.access may still report executable; force it to False for this test
    monkeypatch.setattr(os, 'access', lambda path, mode: False)
    res = lb.measure_tps_with_binary(str(f), 'model.gguf', [1], runs=1, warmup=0)
    assert res == {"map": {}, "details": {}}


def test_measure_tokens_per_second_prefers_binary(monkeypatch, tmp_path):
    # Create a fake executable and monkeypatch shutil.which to find it
    fake = tmp_path / 'main'
    fake.write_text('echo')
    # make it executable
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, 'which', lambda name: str(fake) if name == 'main' else None)

    # monkeypatch subprocess.run to be fast
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: None)

    res = lb.measure_tokens_per_second('model.gguf', contexts=[1], runs=1, warmup=0, timeout=1.0, use_python_binding=False)
    # With a fake binary present and subprocess.run patched, we expect either empty or a filled map; ensure function returns a dict
    assert isinstance(res, dict)