from typing import Tuple, Optional, Callable, Sequence
from .config import VRAMConfig, DEFAULT_CONFIG
from math import log
from .hw import get_gpu_bandwidth_gbps, estimate_free_vram_bytes
import statistics
import time


def calculate_vram_usage(
    model_size_mb: float, 
    n_layers: int, 
    n_gpu_layers: int, 
    context_length: int,
    config: VRAMConfig = DEFAULT_CONFIG
) -> float:
    """
    Calculate VRAM usage based on the existing approximation.

    VRAM = Model_VRAM + Context_VRAM + Overhead

    Model_VRAM = (model_size_mb * gpu_layers_ratio)
    Context_VRAM = context_length * hidden_size * 2 * bytes_per_element / (1024*1024)
    Overhead = configurable MB for llama.cpp operations
    """
    gpu_layers_ratio = min(n_gpu_layers / n_layers, 1.0) if n_layers > 0 else 0.8

    model_vram = model_size_mb * gpu_layers_ratio

    # Context cache VRAM (configurable formula)
    context_vram = (context_length * config.hidden_size * 2 * config.bytes_per_element) / (1024 * 1024)

    total_vram = model_vram + context_vram + config.vram_overhead_mb
    return total_vram


def calculate_ram_usage(
    model_size_mb: float, 
    n_layers: int, 
    n_gpu_layers: int, 
    context_length: int,
    config: VRAMConfig = DEFAULT_CONFIG
) -> float:
    """
    Calculate RAM usage for CPU layers and system overhead

    RAM = Model_RAM + Context_RAM + System_Overhead

    Model_RAM = (model_size_mb * cpu_layers_ratio)
    Context_RAM = context_length * hidden_size * 2 * bytes_per_element / (1024*1024)
    System_Overhead = configurable MB for llama.cpp + OS overhead
    """
    cpu_layers = n_layers - n_gpu_layers
    cpu_layers_ratio = cpu_layers / n_layers if n_layers > 0 else 0.2

    model_ram = model_size_mb * cpu_layers_ratio

    context_ram = (context_length * config.hidden_size * 2 * config.bytes_per_element) / (1024 * 1024)

    total_ram = model_ram + context_ram + config.ram_overhead_mb
    return total_ram


def calculate_total_memory_usage(
    model_size_mb: float, 
    n_layers: int, 
    n_gpu_layers: int, 
    context_length: int,
    config: VRAMConfig = DEFAULT_CONFIG
) -> Tuple[float, float]:
    """Calculate combined VRAM + RAM usage"""
    vram_usage = calculate_vram_usage(model_size_mb, n_layers, n_gpu_layers, context_length, config)
    ram_usage = calculate_ram_usage(model_size_mb, n_layers, n_gpu_layers, context_length, config)
    return vram_usage, ram_usage


def _estimate_tps(
    context_length: int,
    n_layers: int,
    hidden_size: int,
    bytes_per_element: int,
    bw_gbps: float,
    beta: float = 2.0,
    model_meta: Optional[dict] = None,
) -> float:
    """
    Estimate tokens-per-second (sustained) using a KV-aware per-token model.

    We compute bytes_per_token from a GQA/KV-aware formula (via _per_token_kv_bytes).
    This avoids multiplying by the entire context length which previously over-penalized
    larger contexts. If metadata is missing, fall back to a conservative per-token
    estimate based on hidden size and layer count.

    TPS = BW_bytes_per_sec / bytes_per_token
    """
    if context_length <= 0 or bw_gbps <= 0:
        return 0.0

    # Try to compute KV bytes per token from model metadata (preferred)
    try:
        per_token = _per_token_kv_bytes(model_meta or {"n_layers": n_layers, "hidden_size": hidden_size}, bytes_per_elem_override=bytes_per_element)
    except Exception:
        per_token = 0

    # If we couldn't compute a sensible per-token KV size, fall back to a conservative heuristic
    if not per_token or per_token <= 0:
        # Use a heuristic per-token bytes: layers * hidden_size * bytes_per_element
        per_token = max(1, n_layers) * max(1, hidden_size) * max(1, bytes_per_element)

    bytes_per_token = float(beta) * float(per_token)
    bw_bps = bw_gbps * 1e9
    tps = bw_bps / bytes_per_token
    return float(tps)


def _per_token_kv_bytes(model_meta: dict, bytes_per_elem_override: Optional[int] = None) -> int:
    """
    Compute KV bytes per token using GQA-aware formula:
      KV_per_token = 2 * n_kv_heads * head_dim * n_layers * bytes(kv_dtype)

    model_meta should include: n_kv_heads, head_dim, n_layers, kv_dtype (e.g., 'float16')
    Falls back to hidden_size based approximation when needed.
    Returns bytes per token (int).
    """
    if not model_meta:
        return 0

    n_layers = int(model_meta.get("n_layers", model_meta.get("layers", 0)))
    # prefer explicit fields
    n_kv_heads = model_meta.get("n_kv_heads")
    head_dim = model_meta.get("head_dim")
    hidden_size = model_meta.get("hidden_size")
    n_heads = model_meta.get("n_heads")

    # bytes per element
    kv_dtype = model_meta.get("kv_dtype") or model_meta.get("dtype") or "float16"
    if bytes_per_elem_override:
        bpe = int(bytes_per_elem_override)
    else:
        if kv_dtype.lower() in ("float16", "fp16"):
            bpe = 2
        elif kv_dtype.lower() in ("float32", "fp32"):
            bpe = 4
        elif kv_dtype.lower() in ("int8",):
            bpe = 1
        else:
            bpe = 2

    if n_kv_heads is None or head_dim is None:
        # try to infer from hidden_size and n_heads
        if hidden_size and n_heads:
            head_dim = head_dim or (hidden_size // n_heads)
            n_kv_heads = n_kv_heads or n_heads
        elif hidden_size:
            # fallback: assume n_kv_heads == n_heads ==  (hidden_size / 64)
            # conservative default: head_dim = 64
            head_dim = head_dim or 64
            n_kv_heads = n_kv_heads or max(1, hidden_size // head_dim)

    # final sanity defaults
    n_layers = max(1, int(n_layers or 1))
    n_kv_heads = max(1, int(n_kv_heads or 1))
    head_dim = max(1, int(head_dim or 64))

    per_token = 2 * n_kv_heads * head_dim * n_layers * bpe
    return int(per_token)


def fit_latency_model(
    measure_func: Callable[[int], float],
    contexts: Sequence[int] = (0, 512, 2048),
    runs_per_point: int = 3,
) -> tuple[float, float, float]:
    """
    Fit a simple linear latency model latency_ms = a + b * C_eff using measurements.
    Returns (a_ms, b_ms_per_ctx, r2) - r2 is coefficient of determination for fit.
    measure_func(context_len) -> avg_ms_per_token
    """
    xs = []
    ys = []
    for c in contexts:
        samples = []
        for _ in range(runs_per_point):
            try:
                t = float(measure_func(c))
                samples.append(t)
            except Exception:
                continue
        if not samples:
            continue
        xs.append(c)
        ys.append(statistics.mean(samples))

    if len(xs) < 2:
        # insufficient data; return conservative defaults
        return 5.0, 0.002, 0.0

    # simple linear least squares for y = a + b*x
    n = len(xs)
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n)) or 1.0
    b = num / den
    a = y_mean - b * x_mean

    # compute R^2
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    r2 = 0.0
    if ss_tot > 0:
        r2 = max(0.0, 1.0 - (ss_res / ss_tot))

    return float(a), float(b), float(r2)


def validate_context(measure_func: Optional[Callable[[int], bool]], candidate_C: int, attempts: int = 3, step: float = 0.15) -> tuple[int, int, bool]:
    """
    Validate candidate context by calling measure_func(C) which should attempt a prefill+1 generation and return True on success.
    If measure_func is None, this is a best-effort pass that returns success=True (no runtime check).
    On failure, reduce C by `step` fraction and retry up to attempts times.

    Returns (final_C, tries, success)
    """
    if measure_func is None:
        return candidate_C, 0, True

    C = int(candidate_C)
    for i in range(1, attempts + 1):
        try:
            ok = bool(measure_func(C))
        except Exception:
            ok = False
        if ok:
            return C, i, True
        # reduce
        C = max(1, int(C * (1.0 - step)))

    return C, attempts, False


def calculate_semantic_throughput_best_context(
    model_size_mb: float,
    n_layers: int,
    n_gpu_layers: int,
    available_vram_mb: int,
    available_ram_mb: Optional[int] = None,
    config: VRAMConfig = DEFAULT_CONFIG,
    bw_gbps: Optional[float] = None,
    measured_bandwidth: bool = False,
    c_ref: int = 32768,
    beta: float = 2.0,
    memory_penalty_pow: float = 2.0,
    measured_tps_map: Optional[dict] = None,
    measured_k: Optional[float] = None,
    model_meta: Optional[dict] = None,
    opts: Optional[dict] = None,
) -> dict:
    """
    Choose best context by maximizing semantic throughput (tokens/sec * usefulness).

    Returns a diagnostics dict with chosen context, TPS, score and candidate list.
    """
    common_sizes = [
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
    ]
    model_meta = model_meta or {}
    opts = opts or {}

    # attempt to resolve free/available VRAM using probes unless opts forces heuristic
    mode = (opts or {}).get("mode") if opts else None
    free_bytes = None
    basis = None
    safety_used = config.vram_safety_margin
    try:
        fb, basis, safety = estimate_free_vram_bytes(mode=(mode or "auto"), free_vram_override=(opts or {}).get("free_vram_mb"), probes=(opts or {}).get("free_vram_probes"))
        # convert to MB for backward compatibility
        usable_vram = (fb // (1024 * 1024)) * safety
        free_bytes = fb
        safety_used = safety
    except Exception:
        usable_vram = available_vram_mb * config.vram_safety_margin

    # Resolve bandwidth if not provided
    if bw_gbps is None:
        bw_gbps = get_gpu_bandwidth_gbps(available_vram_mb=available_vram_mb, measured=measured_bandwidth)

    alloc_backoff_attempts = opts.get("_alloc_backoff_attempts", 0)
    candidates = []
    for C in common_sizes:
        # Memory feasibility
        # prefer GQA-aware KV math if model_meta is provided
        if model_meta:
            per_token = _per_token_kv_bytes({**model_meta, "n_layers": n_layers}, config.bytes_per_element)
            batch_size = (opts or {}).get("batch_size", 1)
            overhead_frac = (opts or {}).get("kv_paging_overhead_frac", 0.10)
            kv_bytes = C * per_token * batch_size * (1.0 + overhead_frac)
            # model weight bytes if provided
            wb = model_meta.get("weight_bytes_vram")
            if wb is None:
                weight_bytes = int(model_size_mb * 1024 * 1024)
            else:
                weight_bytes = int(wb)
            vram_used_bytes = int(weight_bytes + kv_bytes + config.vram_overhead_mb * 1024 * 1024)
            vram_used = vram_used_bytes / (1024 * 1024)
        else:
            vram_used = calculate_vram_usage(model_size_mb, n_layers, n_gpu_layers, C, config)
        if vram_used > usable_vram:
            continue

        if available_ram_mb is not None:
            ram_used = calculate_ram_usage(model_size_mb, n_layers, n_gpu_layers, C, config)
            if ram_used > available_ram_mb * config.ram_safety_margin:
                continue

        # If we have measured TPS map, prefer it (or use fitted k)
        if measured_tps_map and C in measured_tps_map:
            tps = float(measured_tps_map[C])
        elif measured_k:
            # fitted model TPS ~= k / (C + eps)
            eps = 1.0
            tps = float(measured_k / (C + eps))
        else:
            # use model_meta if available for a more accurate per-token bytes estimate
            try:
                tps = _estimate_tps(C, n_layers, config.hidden_size, config.bytes_per_element, bw_gbps, beta=beta, model_meta=model_meta)
            except TypeError:
                # fallback for safety
                tps = _estimate_tps(C, n_layers, config.hidden_size, config.bytes_per_element, bw_gbps, beta=beta)

        # usefulness per token (diminishing returns). Simpler: normalized 1/(1 + C/c_ref)
        usefulness = 1.0 / (1.0 + (C / max(1, c_ref)))

        mem_margin = 1.0 - min(1.0, vram_used / max(1e-6, usable_vram))

        score = tps * usefulness * (mem_margin ** memory_penalty_pow)

        candidates.append({
            "context": C,
            "vram_used_mb": vram_used,
            "tps": tps,
            "usefulness": usefulness,
            "mem_margin": mem_margin,
            "score": score,
        })

    if not candidates:
        return {
            "chosen": 0,
            "bw_gbps": bw_gbps,
            "candidates": [],
            "reason": "no feasible context fits memory constraints",
        }

    # pick top by score
    best = max(candidates, key=lambda x: x["score"])

    chosen = int(best["context"])

    # Run allocation backoff validation if requested (default True)
    verify_alloc = (opts or {}).get("verify_alloc", True)
    alloc_step = (opts or {}).get("alloc_backoff_step", 0.15)
    alloc_tries = (opts or {}).get("alloc_backoff_tries", 3)
    measure_func = (opts or {}).get("measure_func")
    final_C, attempts, ok = validate_context(measure_func, chosen, attempts=alloc_tries, step=alloc_step)

    reason = "memory"
    if not ok and final_C < chosen:
        reason = "alloc_backoff"
    result = {
        "chosen": final_C,
        "bw_gbps": float(bw_gbps),
        "tps": float(best["tps"]),
        "score": float(best["score"]),
        "candidates": candidates,
        "reason": reason,
        "available_bytes_basis": basis,
        "available_bytes_used": free_bytes,
        "safety_used": safety_used,
        "alloc_backoff_attempts": attempts,
    }

    # build debug payload with detailed fields when requested
    if opts.get("debug"):
        # attempt to reconstruct per-token and weight info for the chosen context
        dbg_per_token = None
        dbg_kv_dtype = None
        dbg_n_kv_heads = None
        dbg_head_dim = None
        dbg_n_layers = n_layers
        dbg_batch_size = (opts or {}).get("batch_size", 1)
        dbg_overhead_frac = (opts or {}).get("kv_paging_overhead_frac", 0.10)
        dbg_weight_bytes = None
        dbg_mem_bound_context = None

        if model_meta:
            dbg_per_token = _per_token_kv_bytes({**model_meta, "n_layers": n_layers}, config.bytes_per_element)
            dbg_kv_dtype = model_meta.get("kv_dtype") or model_meta.get("dtype")
            dbg_n_kv_heads = model_meta.get("n_kv_heads") or model_meta.get("n_heads")
            # safe compute head_dim: prefer explicit, else infer from hidden_size and n_heads when present
            n_heads_val = model_meta.get("n_heads") or 1
            hidden_val = model_meta.get("hidden_size")
            if model_meta.get("head_dim"):
                dbg_head_dim = model_meta.get("head_dim")
            elif hidden_val:
                try:
                    dbg_head_dim = int(hidden_val) // int(n_heads_val)
                except Exception:
                    dbg_head_dim = None
            else:
                dbg_head_dim = None
            wb = model_meta.get("weight_bytes_vram")
            dbg_weight_bytes = int(wb) if wb is not None else int(model_size_mb * 1024 * 1024)
            # compute mem-bound max context given available bytes
            try:
                usable_bytes = int((free_bytes or (available_vram_mb * 1024 * 1024)) * safety_used)
                dbg_mem_bound_context = int(max(0, (usable_bytes - dbg_weight_bytes - int(config.vram_overhead_mb * 1024 * 1024)) // (dbg_per_token * dbg_batch_size * (1.0 + dbg_overhead_frac))))
            except Exception:
                dbg_mem_bound_context = None

        debug = {
            "available_bytes_basis": basis,
            "available_bytes_used": free_bytes,
            "safety_used": safety_used,
            "weight_bytes_vram_used": dbg_weight_bytes,
            "kv_dtype": dbg_kv_dtype,
            "n_kv_heads": dbg_n_kv_heads,
            "head_dim": dbg_head_dim,
            "n_layers": dbg_n_layers,
            "batch_size": dbg_batch_size,
            "overhead_frac_used": dbg_overhead_frac,
            "mem_bound_context": dbg_mem_bound_context,
            "latency_bound_context": None,
            "reason": reason,
            "alloc_backoff_attempts": attempts,
        }
        result["debug"] = debug

    # compute vram_used for the final_C (may differ from the candidate best['context'] when alloc backoff occurred)
    try:
        if model_meta:
            per_token = _per_token_kv_bytes({**model_meta, "n_layers": n_layers}, config.bytes_per_element)
            batch_size = (opts or {}).get("batch_size", 1)
            overhead_frac = (opts or {}).get("kv_paging_overhead_frac", 0.10)
            kv_bytes = final_C * per_token * batch_size * (1.0 + overhead_frac)
            wb = model_meta.get("weight_bytes_vram")
            if wb is None:
                weight_bytes = int(model_size_mb * 1024 * 1024)
            else:
                weight_bytes = int(wb)
            vram_used_bytes_final = int(weight_bytes + kv_bytes + config.vram_overhead_mb * 1024 * 1024)
            vram_used_mb_final = vram_used_bytes_final / (1024 * 1024)
        else:
            vram_used_mb_final = calculate_vram_usage(model_size_mb, n_layers, n_gpu_layers, final_C, config)
    except Exception:
        vram_used_mb_final = None

    result["vram_used_mb"] = vram_used_mb_final

    return result


def calculate_max_context(
    model_size_mb: float,
    n_layers: int,
    n_gpu_layers: int,
    available_vram_mb: int,
    available_ram_mb: Optional[int] = None,
    config: VRAMConfig = DEFAULT_CONFIG
) -> int:
    """Calculate maximum context length for given VRAM and RAM constraints

    Accepts optional model_meta and opts via config if provided in config.extra (backwards compatible).
    """
    # Backwards compatible: allow model_meta and opts passed via config.extra dict (rare),
    # but prefer explicit access via kwargs if updated in future.

    # Try to get a free VRAM estimate via probe/heuristic
    try:
        fb, basis, safety = estimate_free_vram_bytes()
        usable_vram_bytes = int(fb * safety)
        usable_vram = usable_vram_bytes / (1024 * 1024)
    except Exception:
        usable_vram = available_vram_mb * config.vram_safety_margin

    # If model_meta present in config.extra, use it; keep current formula otherwise
    model_meta = getattr(config, "model_meta", None) or None
    opts = getattr(config, "opts", None) or None

    if model_meta:
        per_token = _per_token_kv_bytes({**model_meta, "n_layers": n_layers}, config.bytes_per_element)
        batch_size = (opts or {}).get("batch_size", 1)
        overhead_frac = (opts or {}).get("kv_paging_overhead_frac", 0.10)
        wb = model_meta.get("weight_bytes_vram")
        if wb is None:
            weight_bytes = int(model_size_mb * 1024 * 1024)
        else:
            weight_bytes = int(wb)
        # available for kv in bytes
        available_for_kv_bytes = max(0, int(usable_vram * 1024 * 1024) - weight_bytes - int(config.vram_overhead_mb * 1024 * 1024))
        if available_for_kv_bytes <= 0:
            return 0
        max_context = int(available_for_kv_bytes // (per_token * batch_size * (1.0 + overhead_frac)))
    else:
        usable_vram = usable_vram
        gpu_layers_ratio = min(n_gpu_layers / n_layers, 1.0) if n_layers > 0 else 0.8
        model_vram = model_size_mb * gpu_layers_ratio
        context_vram_budget = usable_vram - model_vram - config.vram_overhead_mb
        if context_vram_budget <= 0:
            return 0
        max_context = int((context_vram_budget * 1024 * 1024) / (config.hidden_size * 2 * config.bytes_per_element))

    # If RAM constraint present, compute RAM-based max
    if available_ram_mb is not None and not model_meta:
        usable_ram = available_ram_mb * config.ram_safety_margin
        cpu_layers_ratio = (n_layers - n_gpu_layers) / n_layers if n_layers > 0 else 0.2
        model_ram = model_size_mb * cpu_layers_ratio
        context_ram_budget = usable_ram - model_ram - config.ram_overhead_mb
        if context_ram_budget > 0:
            ram_max_context = int((context_ram_budget * 1024 * 1024) / (config.hidden_size * 2 * config.bytes_per_element))
            max_context = min(max_context, ram_max_context)

    # Round down to nearest common context sizes
    common_sizes = [
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
    ]
    for size in reversed(common_sizes):
        if max_context >= size:
            return size

    return 512  # Minimum reasonable context