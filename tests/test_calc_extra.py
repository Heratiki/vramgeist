from src.vramgeist.calc import (
    _estimate_tps,
    calculate_semantic_throughput_best_context,
)


def test_estimate_tps_basic():
    tps = _estimate_tps(context_length=1024, n_layers=32, hidden_size=4096, bytes_per_element=2, bw_gbps=100.0, beta=2.0)
    assert isinstance(tps, float)
    assert tps > 0.0


def test_calculate_semantic_throughput_with_measured_map():
    measured_map = {1024: 100.0, 2048: 60.0, 4096: 30.0}
    res = calculate_semantic_throughput_best_context(
        model_size_mb=2000,
        n_layers=32,
        n_gpu_layers=32,
        available_vram_mb=8192,
        available_ram_mb=16384,
        measured_tps_map=measured_map,
    )
    assert isinstance(res, dict)
    chosen = res.get("chosen")
    assert isinstance(chosen, int)
    # If a non-zero context was chosen, it should appear in the candidates list
    if chosen != 0:
        assert any(c.get("context") == chosen for c in res.get("candidates", []))


def test_calculate_semantic_throughput_no_fit_returns_zero():
    # Make model huge so no context fits
    res = calculate_semantic_throughput_best_context(
        model_size_mb=20000,
        n_layers=32,
        n_gpu_layers=32,
        available_vram_mb=8192,
        available_ram_mb=16384,
    )
    assert isinstance(res, dict)
    assert res["chosen"] == 0
