import os
import tempfile

from src.vramgeist.calc import calculate_semantic_throughput_best_context
from src.vramgeist.ui import analyze_gguf_file_with_config


def test_selection_uses_measured_tps_map():
    # Baseline analytic estimate with low BW (prefers larger contexts analytically)
    baseline = calculate_semantic_throughput_best_context(
        model_size_mb=4000,
        n_layers=32,
        n_gpu_layers=32,
        available_vram_mb=24000,
        available_ram_mb=64000,
        bw_gbps=64.0,
    )

    # Provide measured TPS that strongly favors a small context (512)
    measured_map = {512: 800.0, 1024: 300.0, 4096: 60.0, 8192: 20.0}

    measured = calculate_semantic_throughput_best_context(
        model_size_mb=4000,
        n_layers=32,
        n_gpu_layers=32,
        available_vram_mb=24000,
        available_ram_mb=64000,
        bw_gbps=64.0,
        measured_tps_map=measured_map,
    )

    # Measured-driven choice should be the small context (512)
    assert int(measured["chosen"]) == 512
    # And it should differ from baseline if baseline wasn't 512
    assert int(baseline["chosen"]) != int(measured["chosen"]) or int(baseline["chosen"]) == 512


def test_ui_invokes_bench(monkeypatch, tmp_path):
    called = {"flag": False}

    def fake_measure_tokens_per_second(model_path, contexts, **kwargs):
        called["flag"] = True
        # return a simple mapping
    return {"map": {1024: 100.0, 4096: 40.0}, "details": {}}

    # Patch the symbol imported into the UI module (analyze_gguf_file_with_config imports it)
    monkeypatch.setattr("src.vramgeist.ui.measure_tokens_per_second", fake_measure_tokens_per_second)

    # Create a tiny temp file to act as model path (size-based functions will work)
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF")

    # Call analyzer with measure_tps=True and ensure our fake was invoked
    res = analyze_gguf_file_with_config(str(f), measure_tps=True)

    assert called["flag"] is True
    # result should be a dict and contain recommendation
    assert "recommendation" in res
