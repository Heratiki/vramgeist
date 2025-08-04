from typing import Tuple, Optional
from .config import VRAMConfig, DEFAULT_CONFIG


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