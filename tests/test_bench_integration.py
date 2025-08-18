import os
import pytest

from src.vramgeist.bench.llama_bench import measure_tokens_per_second


@pytest.mark.skipif(
    os.environ.get("VRAMGEIST_RUN_SLOW_BENCH", "0") != "1",
    reason="Slow bench disabled; set VRAMGEIST_RUN_SLOW_BENCH=1 to enable",
)
def test_real_bench_runs_quick_smoke():
    """Gated integration test: run a very small bench against a provided model path.

    This test is gated and should be enabled manually by setting two env vars:
      - VRAMGEIST_RUN_SLOW_BENCH=1
      - VRAMGEIST_BENCH_MODEL=/path/to/model.gguf

    It performs a short bench (1 run per context) to smoke test the bench path.
    """
    model = os.environ.get("VRAMGEIST_BENCH_MODEL")
    assert model, "Set VRAMGEIST_BENCH_MODEL to a GGUF model path to run this test"

    # Use small contexts and 1 run to keep it short
    res = measure_tokens_per_second(model, contexts=[512, 1024], n_predict=32, runs=1, warmup=0, timeout=30.0)

    # Expect the richer structure with map and details
    assert isinstance(res, dict)
    assert "map" in res and "details" in res
    # Map should be present but may be empty if the binary/binding failed
    assert isinstance(res["map"], dict)
    assert isinstance(res["details"], dict)
