"""
Lightweight benchmarking helpers to measure tokens/sec using llama-cpp-python when available,
with a subprocess fallback to a llama.cpp binary if provided.

This module intentionally keeps external dependencies optional and falls back gracefully.
"""
from __future__ import annotations
import time
import shlex
import subprocess
import tempfile
import os
import shutil
from typing import List, Dict, Optional, Tuple, Any


def _make_repeated_prompt(token_word: str, approx_tokens: int) -> str:
    # Simple prompt with repeated short tokens; not perfect tokenization but fast and deterministic
    return " ".join([token_word] * max(1, int(approx_tokens)))


def measure_tps_with_python_binding(
    model_path: str,
    contexts: List[int],
    n_predict: int = 128,
    runs: int = 3,
    warmup: int = 1,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Attempt to measure tokens/sec using the `llama_cpp` Python binding.
    Returns a dict with keys:
      - map: {context: tps}
      - details: {context: {"runs": [durations], "elapsed": total_elapsed, "success": bool}}
    On error returns an empty shape with empty map/details.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        return {"map": {}, "details": {}}

    # instantiate model
    try:
        llm = Llama(model_path=model_path)
    except Exception:
        return {"map": {}, "details": {}}

    results_map: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}

    for C in contexts:
        prompt = _make_repeated_prompt("Hello", C)

        # warmup
        try:
            for _ in range(warmup):
                _ = llm.create(prompt=prompt, max_tokens=n_predict)
        except Exception:
            # If generation fails during warmup, skip this context
            continue

        # timed runs: record per-run durations
        run_durations: List[float] = []
        success = True
        t_start = time.perf_counter()
        for _ in range(runs):
            try:
                r0 = time.perf_counter()
                _ = llm.create(prompt=prompt, max_tokens=n_predict)
                r1 = time.perf_counter()
                run_durations.append(r1 - r0)
            except Exception:
                success = False
                break
        if not success or not run_durations:
            # skip this context if any run failed
            details[C] = {"runs": run_durations, "elapsed": None, "success": False}
            continue
        elapsed = time.perf_counter() - t_start
        if elapsed <= 0:
            details[C] = {"runs": run_durations, "elapsed": elapsed, "success": False}
            continue
        tps = (n_predict * len(run_durations)) / elapsed
        results_map[C] = tps
        details[C] = {"runs": run_durations, "elapsed": elapsed, "success": True}

    return {"map": results_map, "details": details}


def measure_tps_with_binary(
    llama_bin: str,
    model_path: str,
    contexts: List[int],
    n_predict: int = 128,
    runs: int = 3,
    warmup: int = 1,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Measure tokens/sec by invoking an external llama.cpp binary. The exact CLI flags vary by build;
    we use a simple, common invocation: `./main -m model.gguf --n_predict N --prompt "..."` and measure time.

    Returns structure {"map": {context: tps}, "details": {context: {...}}}.
    If the binary is not executable or invocation fails, returns {}.
    """
    if not os.path.exists(llama_bin) or not os.access(llama_bin, os.X_OK):
        return {"map": {}, "details": {}}

    results_map: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}

    for C in contexts:
        prompt = _make_repeated_prompt("Hello", C)

        # Warmup
        try:
            for _ in range(warmup):
                cmd = f'{shlex.quote(llama_bin)} -m {shlex.quote(model_path)} --n_predict {n_predict} --prompt {shlex.quote(prompt)}'
                subprocess.run(cmd, shell=True, timeout=min(30.0, timeout))
        except subprocess.SubprocessError:
            continue

        # timed runs with per-run durations
        run_durations: List[float] = []
        success = True
        t_start = time.perf_counter()
        try:
            for _ in range(runs):
                cmd = f'{shlex.quote(llama_bin)} -m {shlex.quote(model_path)} --n_predict {n_predict} --prompt {shlex.quote(prompt)}'
                r0 = time.perf_counter()
                subprocess.run(cmd, shell=True, timeout=timeout)
                r1 = time.perf_counter()
                run_durations.append(r1 - r0)
        except subprocess.SubprocessError:
            success = False

        if not success or not run_durations:
            details[C] = {"runs": run_durations, "elapsed": None, "success": False}
            continue
        elapsed = time.perf_counter() - t_start
        if elapsed <= 0:
            details[C] = {"runs": run_durations, "elapsed": elapsed, "success": False}
            continue
        tps = (n_predict * len(run_durations)) / elapsed
        results_map[C] = tps
        details[C] = {"runs": run_durations, "elapsed": elapsed, "success": True}
    return {"map": results_map, "details": details}


def measure_tokens_per_second(
    model_path: str,
    contexts: List[int] = [1024, 4096, 8192],
    n_predict: int = 128,
    runs: int = 3,
    warmup: int = 1,
    timeout: float = 120.0,
    use_python_binding: bool = True,
    llama_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper: prefer a llama.cpp binary measurement first (if provided or discoverable),
    then try the python binding (`llama_cpp`) as a fallback. Returns dict context->tps or empty dict on failure.
    """
    # Build candidate binaries list (user-provided, env, discover on PATH)
    bins_to_try: List[str] = []
    env_bin = os.environ.get("LLAMA_CPP_BIN")
    if llama_bin:
        bins_to_try.append(llama_bin)
    if env_bin:
        bins_to_try.append(env_bin)

    for candidate in ("main", "main.exe"):
        path = shutil.which(candidate)
        if path:
            bins_to_try.append(path)
            break

    # Try binaries first
    for b in bins_to_try:
        try:
            res = measure_tps_with_binary(b, model_path, contexts, n_predict, runs, warmup, timeout)
            if res and res.get("map"):
                return res
        except Exception:
            continue

    # Next try python binding if allowed
    if use_python_binding:
        res = measure_tps_with_python_binding(model_path, contexts, n_predict, runs, warmup, timeout)
        if res and res.get("map"):
            return res

    return {"map": {}, "details": {}}


def fit_k_from_measurements(measurements: Any, eps: float = 1.0) -> Optional[float]:
    """
    Fit a simple model TPS(C) ≈ k / (C + eps) using least-squares on measured points.
    Return k or None if not enough data.
    """
    # measurements may be either a map {C: tps} or the richer structure {"map": {...}}
    if not measurements:
        return None
    if isinstance(measurements, dict) and "map" in measurements:
        data = measurements["map"]
    else:
        data = measurements

    ks = []
    for C, tps in data.items():
        try:
            c_int = int(C)
            tps_f = float(tps)
        except Exception:
            continue
        if tps_f <= 0:
            continue
        ks.append(tps_f * (c_int + eps))
    if not ks:
        return None
    ks.sort()
    return ks[len(ks) // 2]
