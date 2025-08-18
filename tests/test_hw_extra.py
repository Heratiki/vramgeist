import re
from src.vramgeist.hw import (
    _lookup_bandwidth_by_name,
    get_gpu_bandwidth_gbps,
    _parse_dxdiag_content,
    _clear_gpu_cache,
)


def test_lookup_bandwidth_by_name_matches():
    assert _lookup_bandwidth_by_name("NVIDIA RTX 4090") == 1008.0
    assert _lookup_bandwidth_by_name("a100") == 1555.0
    assert _lookup_bandwidth_by_name("unknowngpu") is None


def test_get_gpu_bandwidth_gbps_heuristic():
    # Provide available_vram_mb to exercise heuristic branch
    bw = get_gpu_bandwidth_gbps(gpu_name=None, measured=False, available_vram_mb=16384)
    # Heuristic: (available_vram_mb/1024)*80 clamped
    assert bw >= 20.0
    assert bw <= 2000.0


def test_parse_dxdiag_content_tiers():
    # Tier A dedicated memory
    content_a = "Dedicated memory: 8192 MB\nSome other lines"
    vals_a = _parse_dxdiag_content(content_a)
    assert isinstance(vals_a, list)
    assert 8192 in vals_a

    # Tier B video memory
    content_b = "Video Memory: 4096 MB\nDisplay Memory: 4096 MB"
    vals_b = _parse_dxdiag_content(content_b)
    assert 4096 in vals_b

    # Tier C approximate total
    content_c = "Total memory: 16384 MB\nApprox Total Memory: 16 GB"
    vals_c = _parse_dxdiag_content(content_c)
    assert any(v >= 16000 for v in vals_c)


def test_clear_gpu_cache_idempotent():
    _clear_gpu_cache()
    # calling again should not raise
    _clear_gpu_cache()
