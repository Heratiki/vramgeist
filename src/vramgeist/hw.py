import platform
import subprocess
from typing import Tuple, Optional
from rich.console import Console

console = Console(force_terminal=True, width=120)


def get_gpu_memory(timeout: float = 2.0, policy: str = "max") -> int:
    """
    Return selected GPU total VRAM in MB using nvidia-smi with a timeout.
    Policy:
      - "first": pick the first visible GPU
      - "max": pick the GPU with maximum total VRAM
    Fallback to 8192 MB if unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            console.print(f"[yellow]nvidia-smi error: {result.stderr.strip() or 'unknown error'}. Using default 8GB VRAM.[/yellow]")
            return 8192

        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if not lines:
            console.print("[yellow]nvidia-smi returned no GPUs. Using default 8GB VRAM.[/yellow]")
            return 8192

        try:
            values = [int(x) for x in lines]
        except ValueError:
            console.print("[yellow]Failed to parse nvidia-smi output. Using default 8GB VRAM.[/yellow]")
            return 8192

        if policy == "first":
            return values[0]
        # default "max"
        return max(values)
    except FileNotFoundError:
        console.print("[yellow]nvidia-smi not found. Using default 8GB VRAM.[/yellow]")
        return 8192
    except subprocess.TimeoutExpired:
        console.print("[yellow]nvidia-smi timed out. Using default 8GB VRAM.[/yellow]")
        return 8192
    except Exception as e:
        console.print(f"[yellow]Unexpected GPU probe error: {e}. Using default 8GB VRAM.[/yellow]")
        return 8192


def get_system_memory() -> Tuple[int, int]:
    """
    Get total and available system RAM in MB.
    Returns (total_mb, available_mb). Fallback to (16384, 12288) on error.
    """
    try:
        import psutil  # import locally to avoid issues if absent in some environments
        memory = psutil.virtual_memory()
        total_ram_mb = memory.total // (1024 * 1024)
        available_ram_mb = memory.available // (1024 * 1024)
        return total_ram_mb, available_ram_mb
    except Exception as e:
        console.print(f"[yellow]Could not detect system RAM: {e}. Using defaults.[/yellow]")
        return 16384, 12288