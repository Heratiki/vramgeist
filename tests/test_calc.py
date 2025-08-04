from src.vramgeist.calc import (
    calculate_vram_usage,
    calculate_ram_usage,
    calculate_max_context,
    calculate_total_memory_usage,
)
from src.vramgeist.config import VRAMConfig


class TestVRAMCalculations:
    """Test VRAM calculation functions"""
    
    def test_calculate_vram_usage_basic(self):
        """Test basic VRAM usage calculation"""
        config = VRAMConfig()
        result = calculate_vram_usage(
            model_size_mb=7000,
            n_layers=32,
            n_gpu_layers=32,  # Full GPU
            context_length=2048,
            config=config
        )
        
        # Should be model + context + overhead
        expected_context = (2048 * 4096 * 2 * 2) / (1024 * 1024)  # ~32MB
        _ = 7000 + expected_context + 500  # expected_total ~7532MB
        
        assert result > 7500
        assert result < 7600
    
    def test_calculate_vram_usage_partial_gpu(self):
        """Test VRAM usage with partial GPU layers"""
        config = VRAMConfig()
        result = calculate_vram_usage(
            model_size_mb=7000,
            n_layers=32,
            n_gpu_layers=16,  # Half GPU
            context_length=2048,
            config=config
        )
        
        # Should use half the model + context + overhead
        expected_model = 7000 * 0.5  # 3500MB
        expected_context = (2048 * 4096 * 2 * 2) / (1024 * 1024)  # ~32MB
        _ = expected_model + expected_context + 500  # expected_total ~4032MB
        
        assert result > 4000
        assert result < 4100
    
    def test_calculate_ram_usage_cpu_only(self):
        """Test RAM usage with CPU-only execution"""
        config = VRAMConfig()
        result = calculate_ram_usage(
            model_size_mb=7000,
            n_layers=32,
            n_gpu_layers=0,  # CPU only
            context_length=2048,
            config=config
        )
        
        # Should use full model + context + overhead
        expected_model = 7000  # Full model on CPU
        expected_context = (2048 * 4096 * 2 * 2) / (1024 * 1024)  # ~32MB  
        _ = expected_model + expected_context + 1000  # expected_total ~8032MB
        
        assert result > 8000
        assert result < 8100
    
    def test_calculate_max_context_limited_vram(self):
        """Test max context calculation with limited VRAM"""
        config = VRAMConfig()
        result = calculate_max_context(
            model_size_mb=4000,  # Smaller model to leave more VRAM budget
            n_layers=32,
            n_gpu_layers=32,
            available_vram_mb=8192,  # 8GB
            available_ram_mb=16384,  # 16GB
            config=config
        )
        
        # With 8GB VRAM, 4GB model should leave room for context
        assert result > 0
        assert result < 200000  # Should be reasonable context length
    
    def test_calculate_max_context_zero_budget(self):
        """Test max context when model exceeds available VRAM"""
        config = VRAMConfig()
        result = calculate_max_context(
            model_size_mb=10000,  # 10GB model
            n_layers=32,
            n_gpu_layers=32,
            available_vram_mb=8192,  # 8GB VRAM
            available_ram_mb=16384,
            config=config
        )
        
        # Model is too large for VRAM
        assert result == 0
    
    def test_calculate_total_memory_usage(self):
        """Test combined VRAM + RAM calculation"""
        config = VRAMConfig()
        vram, ram = calculate_total_memory_usage(
            model_size_mb=7000,
            n_layers=32,
            n_gpu_layers=16,  # Half on GPU
            context_length=2048,
            config=config
        )
        
        assert vram > 0
        assert ram > 0
        # Total should be reasonable for a 7B model
        assert vram + ram > 7000
        assert vram + ram < 10000
    
    def test_config_customization(self):
        """Test that custom config affects calculations"""
        default_config = VRAMConfig()
        custom_config = VRAMConfig(
            hidden_size=8192,  # Double the hidden size
            vram_overhead_mb=1000  # Double overhead
        )
        
        default_result = calculate_vram_usage(
            model_size_mb=7000,
            n_layers=32,
            n_gpu_layers=32,
            context_length=2048,
            config=default_config
        )
        
        custom_result = calculate_vram_usage(
            model_size_mb=7000,
            n_layers=32,
            n_gpu_layers=32,
            context_length=2048,
            config=custom_config
        )
        
        # Custom config should use more VRAM due to larger hidden size and overhead
        assert custom_result > default_result