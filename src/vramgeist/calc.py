from typing import Tuple, Optional
from .config import VRAMConfig, DEFAULT_CONFIG
from math import log
from .hw import get_gpu_bandwidth_gbps


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
) -> float:
    """
    Estimate tokens-per-second (sustained) using a conservative memory-bound per-token model.

    bytes_per_token ~ beta * n_layers * hidden_size * bytes_per_element * context_length
    TPS = (BW_bytes_per_sec) / bytes_per_token
    """
    if context_length <= 0 or bw_gbps <= 0:
        return 0.0

    bytes_per_token = beta * max(1, n_layers) * max(1, hidden_size) * bytes_per_element * max(1, context_length)
    bw_bps = bw_gbps * 1e9
    tps = bw_bps / bytes_per_token
    return float(tps)


def calculate_semantic_throughput_best_context(
    model_size_mb: float,
    n_layers: int,
    n_gpu_layers: int,
    available_vram_mb: int,
    available_ram_mb: Optional[int] = None,
    config: VRAMConfig = DEFAULT_CONFIG,
    bw_gbps: Optional[float] = None,
    measured_bandwidth: bool = False,
    c_ref: int = 8192,
    beta: float = 2.0,
    memory_penalty_pow: float = 2.0,
    measured_tps_map: Optional[dict] = None,
    measured_k: Optional[float] = None,
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
    ]

    usable_vram = available_vram_mb * config.vram_safety_margin

    # Resolve bandwidth if not provided
    if bw_gbps is None:
        bw_gbps = get_gpu_bandwidth_gbps(available_vram_mb=available_vram_mb, measured=measured_bandwidth)

    candidates = []
    for C in common_sizes:
        # Memory feasibility
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

    return {
        "chosen": int(best["context"]),
        "bw_gbps": float(bw_gbps),
        "tps": float(best["tps"]),
        "score": float(best["score"]),
        "candidates": candidates,
    }


def calculate_max_context(
    model_size_mb: float,
    n_layers: int,
    n_gpu_layers: int,
    available_vram_mb: int,
    available_ram_mb: Optional[int] = None,
    config: VRAMConfig = DEFAULT_CONFIG
) -> int:
    """Calculate maximum context length for given VRAM and RAM constraints"""
    usable_vram = available_vram_mb * config.vram_safety_margin

    gpu_layers_ratio = min(n_gpu_layers / n_layers, 1.0) if n_layers > 0 else 0.8
    model_vram = model_size_mb * gpu_layers_ratio

    context_vram_budget = usable_vram - model_vram - config.vram_overhead_mb

    if context_vram_budget <= 0:
        return 0

    vram_max_context = int((context_vram_budget * 1024 * 1024) / (config.hidden_size * 2 * config.bytes_per_element))

    if available_ram_mb is not None:
        usable_ram = available_ram_mb * config.ram_safety_margin

        cpu_layers_ratio = (n_layers - n_gpu_layers) / n_layers if n_layers > 0 else 0.2
        model_ram = model_size_mb * cpu_layers_ratio

        context_ram_budget = usable_ram - model_ram - config.ram_overhead_mb

        if context_ram_budget > 0:
            ram_max_context = int((context_ram_budget * 1024 * 1024) / (config.hidden_size * 2 * config.bytes_per_element))
            max_context = min(vram_max_context, ram_max_context)
        else:
            max_context = vram_max_context
    else:
        max_context = vram_max_context

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