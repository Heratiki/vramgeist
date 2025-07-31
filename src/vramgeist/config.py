from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VRAMConfig:
    """Configuration parameters for VRAM calculations"""
    hidden_size: int = 4096
    bytes_per_element: int = 2  # fp16
    vram_overhead_mb: int = 500
    ram_overhead_mb: int = 1000
    vram_safety_margin: float = 0.9
    ram_safety_margin: float = 0.8
    
    @classmethod
    def from_profile(cls, profile: str) -> "VRAMConfig":
        """Create config from predefined profile"""
        profiles = {
            "default": cls(),
            "conservative": cls(
                vram_safety_margin=0.8,
                ram_safety_margin=0.7,
                vram_overhead_mb=800,
                ram_overhead_mb=1500
            ),
            "aggressive": cls(
                vram_safety_margin=0.95,
                ram_safety_margin=0.9,
                vram_overhead_mb=300,
                ram_overhead_mb=500
            )
        }
        return profiles.get(profile, cls())
    
    def update_from_args(self, args) -> "VRAMConfig":
        """Update config from command line arguments"""
        return VRAMConfig(
            hidden_size=args.hidden_size,
            bytes_per_element=args.bytes_per_element,
            vram_overhead_mb=args.vram_overhead,
            ram_overhead_mb=args.ram_overhead,
            vram_safety_margin=args.vram_safety,
            ram_safety_margin=args.ram_safety
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output"""
        return {
            "hidden_size": self.hidden_size,
            "bytes_per_element": self.bytes_per_element,
            "vram_overhead_mb": self.vram_overhead_mb,
            "ram_overhead_mb": self.ram_overhead_mb,
            "vram_safety_margin": self.vram_safety_margin,
            "ram_safety_margin": self.ram_safety_margin
        }


# Default configuration instance
DEFAULT_CONFIG = VRAMConfig()