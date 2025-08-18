import platform
import subprocess
import tempfile
import os
import re
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    # Optional dev dependency; present during tests or dev environments
    import psutil  # type: ignore
from ._rich_fallback import Console

console = Console(force_terminal=True, width=120)


_GPU_CACHE: dict = {}


def _clear_gpu_cache() -> None:
    """Clear the in-memory GPU detection cache (useful for tests)."""
    _GPU_CACHE.clear()


def _get_psutil():
    """Attempt to import psutil and return the module or None if unavailable."""
    try:
        import importlib
        psutil = importlib.import_module('psutil')
        return psutil
    except Exception:
        return None


def get_gpu_memory(timeout: float = 2.0, policy: str = "max", force: bool = False) -> int:
    """
    Return selected GPU total VRAM in MB with cross-platform detection.
    Supports NVIDIA, AMD, Intel, and Apple GPUs.
    Policy:
      - "first": pick the first visible GPU
      - "max": pick the GPU with maximum total VRAM
    Fallback to 8192 MB if unavailable.
    """
    # Try different GPU detection methods in order of preference
    detectors = [
        _detect_nvidia_gpu,
        _detect_amd_gpu,
        _detect_intel_gpu,
        _detect_apple_gpu,
        _detect_dxdiag_gpu,  # Windows fallback
    ]
    
    # Check cache first (unless force requested)
    cache_key = f"gpu_{policy}"
    if not force and cache_key in _GPU_CACHE:
        try:
            cached = _GPU_CACHE[cache_key]
            return int(cached)
        except Exception:
            pass

    for detector in detectors:
        try:
            result = detector(timeout, policy)
            if result > 0:
                    # Cache result for this run
                    try:
                        _GPU_CACHE[cache_key] = int(result)
                    except Exception:
                        pass
                    return result
        except Exception as e:
            # Only show debug info for unexpected errors, not normal failures
            if "timeout" not in str(e).lower() and "not found" not in str(e).lower():
                console.print(f"[dim yellow]GPU detection method failed: {e}[/dim yellow]")
            continue
    
    console.print("[yellow]No GPU detected. Using default 8GB VRAM.[/yellow]")
    return 8192


def _detect_nvidia_gpu(timeout: float, policy: str) -> int:
    """Detect NVIDIA GPU VRAM using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return 0

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return 0

        values = [int(x) for x in lines]
        console.print(f"[green]Detected NVIDIA GPU(s): {values} MB VRAM[/green]")
        
        if policy == "first":
            return values[0]
        return max(values)
        
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 0


def _detect_amd_gpu(timeout: float, policy: str) -> int:
    """Detect AMD GPU VRAM using rocm-smi"""
    try:
        # Try rocm-smi first (more reliable for newer AMD GPUs)
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            vram_values = []
            for line in lines:
                if "GPU" in line and "vram" in line.lower():
                    # Parse CSV format: GPU[0],vram,Total (B),16106127360
                    parts = line.split(',')
                    if len(parts) >= 4:
                        bytes_value = int(parts[3])
                        mb_value = bytes_value // (1024 * 1024)
                        vram_values.append(mb_value)
            
            if vram_values:
                console.print(f"[green]Detected AMD GPU(s): {vram_values} MB VRAM[/green]")
                return vram_values[0] if policy == "first" else max(vram_values)
        
        # Fallback to radeontop (if available)
        result = subprocess.run(
            ["radeontop", "-d", "-l", "1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and "VRAM" in result.stdout:
            # Parse radeontop output for VRAM info
            # This is less reliable but may work on some systems
            return 0  # Would need specific parsing logic
            
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    
    return 0


def _detect_intel_gpu(timeout: float, policy: str) -> int:
    """Detect Intel GPU VRAM using intel_gpu_top or similar tools"""
    try:
        # Intel GPUs typically share system RAM, so detection is complex
        # intel_gpu_top doesn't directly report VRAM but we can try
        result = subprocess.run(
            ["intel_gpu_top", "-s", "1000", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            # Intel integrated GPUs typically use shared memory
            # Estimate 25% of system RAM as available to GPU (conservative)
            psutil = _get_psutil()
            if psutil is not None:
                try:
                    total_ram = psutil.virtual_memory().total // (1024 * 1024)
                    estimated_vram = min(total_ram // 4, 4096)  # Cap at 4GB
                    console.print(f"[green]Detected Intel GPU: estimated {estimated_vram} MB shared VRAM[/green]")
                    return estimated_vram
                except Exception:
                    pass

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0


def _detect_apple_gpu(timeout: float, policy: str) -> int:
    """Detect Apple Silicon GPU VRAM (M1/M2/M3)"""
    if platform.system() != "Darwin":
        return 0
        
    try:
        # On Apple Silicon, GPU and CPU share unified memory
        # Use system_profiler to get GPU info
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            for line in lines:
                if "Total Number of Cores" in line and "GPU" in line:
                    # This indicates Apple Silicon GPU
                    # Get total system memory as it's unified
                    psutil = _get_psutil()
                    if psutil is not None:
                        try:
                            total_ram = psutil.virtual_memory().total // (1024 * 1024)
                            # Apple Silicon GPUs can use most of system RAM
                            estimated_vram = int(total_ram * 0.7)  # 70% available to GPU
                            console.print(f"[green]Detected Apple Silicon GPU: {estimated_vram} MB unified memory[/green]")
                            return estimated_vram
                        except Exception:
                            pass
                        
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return 0


def _detect_dxdiag_gpu(timeout: float, policy: str) -> int:
    """Detect GPU VRAM using Windows dxdiag (DirectX Diagnostics)"""
    if platform.system() != "Windows":
        return 0
    temp_file = None
    try:
        # Create temporary file for dxdiag output in system temp directory
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"vramgeist_dxdiag_{os.getpid()}.txt")

        # Run dxdiag to generate report (no console output to avoid UI popup)
        result = subprocess.run(
            ["dxdiag", "/t", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout * 3,  # dxdiag can be quite slow
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )

        if result.returncode != 0:
            return 0

        # Wait for file to be written (dxdiag can be slow)
        import time
        for _ in range(10):  # Wait up to 5 seconds
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                break
            time.sleep(0.5)
        else:
            return 0  # File not created or too small

        # Parse the dxdiag output file
        with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        vram_values = _parse_dxdiag_content(content)

        # If dxdiag only surfaced "Total/Display/Approx Total" values but no dedicated/video memory
        # and those values are clearly large shared memory from iGPU, ignore them to avoid overestimation.
        # Heuristic: when only Tier C values are present and they are very large, return empty to force other fallbacks.
        if vram_values and all(v >= 4096 for v in vram_values):
            # Try to detect if any Tier A or B signals exist explicitly; if none, treat as iGPU shared memory case.
            tier_a_or_b_present = False
            # Mirror the patterns from _parse_dxdiag_content to check presence without collecting numbers
            tier_a_check = [
                r'Dedicated\s+memory:\s*\d+?\s*MB',
                r'Dedicated\s+Video\s+Memory:\s*\d+?\s*MB',
                r'VRAM:\s*\d+?\s*MB',
                r'Dedicated\s+Memory:\s*\d+?\s*MB',
                r'Dedicated\s+memory:\s*\d+(?:\.\d+)?\s*GB',
                r'Dedicated\s+Video\s+Memory:\s*\d+(?:\.\d+)?\s*GB',
                r'VRAM:\s*\d+(?:\.\d+)?\s*GB',
                r'Dedicated\s+Memory:\s*\d+(?:\.\d+)?\s*GB',
            ]
            tier_b_check = [
                r'Video\s+memory:\s*\d+?\s*MB',
                r'Video\s+Memory:\s*\d+?\s*MB',
                r'Video\s+memory:\s*\d+(?:\.\d+)?\s*GB',
                r'Video\s+Memory:\s*\d+(?:\.\d+)?\s*GB',
            ]
            for pat in tier_a_check + tier_b_check:
                if re.search(pat, content, re.IGNORECASE):
                    # Check if any match is above threshold 256 MB
                    nums = re.findall(r'(\d+(?:\.\d+)?)', ''.join(re.findall(pat, content, re.IGNORECASE)))
                    for n in nums:
                        try:
                            mb = int(float(n) * (1024 if 'GB' in pat else 1))
                            if mb > 256:
                                tier_a_or_b_present = True
                                break
                        except Exception:
                            continue
                if tier_a_or_b_present:
                    break
            if not tier_a_or_b_present:
                vram_values = []

        if vram_values:
            console.print(f"[green]Detected GPU(s) via dxdiag: {vram_values} MB VRAM[/green]")
            return vram_values[0] if policy == "first" else max(vram_values)

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, PermissionError):
        pass
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass  # File might be locked, ignore cleanup failure

    return 0


def _parse_dxdiag_content(content: str) -> List[int]:
    """Parse dxdiag content to extract GPU VRAM information, prioritizing dedicated VRAM.
    
    Strategy:
      - Tier A (preferred): Dedicated memory fields that represent dedicated VRAM.
      - Tier B: Video memory fields commonly representing VRAM.
      - Tier C: Total or Display memory fields which may include shared memory; used only if A and B are absent.
    Returns a list of unique MB values found within the highest non-empty tier, sorted desc.
    """
    def _collect(patterns: List[str]) -> List[int]:
        vals: List[int] = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    if 'GB' in pattern:
                        vram_mb = int(float(match) * 1024)
                    else:
                        vram_mb = int(match)
                    if vram_mb > 256:
                        vals.append(vram_mb)
                except ValueError:
                    continue
        # unique and sort desc
        return sorted(list(set(vals)), reverse=True)

    # Tier A: Dedicated / VRAM explicit signals
    tier_a_patterns = [
        r'Dedicated\s+memory:\s*(\d+)\s*MB',
        r'Dedicated\s+Video\s+Memory:\s*(\d+)\s*MB',
        r'VRAM:\s*(\d+)\s*MB',
        r'Dedicated\s+Memory:\s*(\d+)\s*MB',

        r'Dedicated\s+memory:\s*(\d+(?:\.\d+)?)\s*GB',
        r'Dedicated\s+Video\s+Memory:\s*(\d+(?:\.\d+)?)\s*GB',
        r'VRAM:\s*(\d+(?:\.\d+)?)\s*GB',
        r'Dedicated\s+Memory:\s*(\d+(?:\.\d+)?)\s*GB',
    ]

    # Tier B: Video memory
    tier_b_patterns = [
        r'Video\s+memory:\s*(\d+)\s*MB',
        r'Video\s+Memory:\s*(\d+)\s*MB',

        r'Video\s+memory:\s*(\d+(?:\.\d+)?)\s*GB',
        r'Video\s+Memory:\s*(\d+(?:\.\d+)?)\s*GB',
    ]

    # Tier C: Total / Display / Memory Size may include shared memory
    tier_c_patterns = [
        r'Approx\.?\s+Total\s+Memory:\s*(\d+)\s*MB',
        r'Total\s+memory:\s*(\d+)\s*MB',
        r'Total\s+Memory:\s*(\d+)\s*MB',
        r'Display\s+memory:\s*(\d+)\s*MB',
        r'Display\s+Memory:\s*(\d+)\s*MB',
        r'Memory\s+Size:\s*(\d+)\s*MB',

        r'Approx\.?\s+Total\s+Memory:\s*(\d+(?:\.\d+)?)\s*GB',
        r'Total\s+memory:\s*(\d+(?:\.\d+)?)\s*GB',
        r'Total\s+Memory:\s*(\d+(?:\.\d+)?)\s*GB',
    ]

    tier_a_vals = _collect(tier_a_patterns)
    if tier_a_vals:
        return tier_a_vals

    tier_b_vals = _collect(tier_b_patterns)
    if tier_b_vals:
        return tier_b_vals

    # Only consider Tier C when there isn't clear evidence that this is an iGPU with tiny/zero dedicated VRAM.
    # If the content explicitly shows Dedicated memory is 0 or very small (<=256 MB) and Video memory is also small,
    # then ignore Tier C to avoid inflating VRAM with shared system memory.
    tier_c_vals = _collect(tier_c_patterns)

    if tier_c_vals:
        # Detect explicit small/zero dedicated memory signal safely (guard group access)
        dedicated_match = re.search(r'Dedicated\s+(?:Video\s+)?Memory:\s*(\d+)\s*MB', content, re.IGNORECASE)
        dedicated_small = False
        if dedicated_match and dedicated_match.group(1):
            try:
                dedicated_small = int(dedicated_match.group(1)) <= 256
            except Exception:
                dedicated_small = False

        # Detect small video memory signal (often iGPU indicator)
        video_match = re.search(r'Video\s+Memory:\s*(\d+)\s*MB', content, re.IGNORECASE)
        video_small = False
        if video_match and video_match.group(1):
            try:
                video_small = int(video_match.group(1)) <= 256
            except Exception:
                video_small = False

        if dedicated_small and video_small:
            return []

    return tier_c_vals


def get_system_memory() -> Tuple[int, int]:
    """
    Get total and available system RAM in MB.
    Returns (total_mb, available_mb). Fallback to (16384, 12288) on error.
    """
    try:
        psutil = _get_psutil()
        if psutil is not None:
            memory = psutil.virtual_memory()
            total_ram_mb = memory.total // (1024 * 1024)
            available_ram_mb = memory.available // (1024 * 1024)
            return total_ram_mb, available_ram_mb
        else:
            raise RuntimeError("psutil not available")
    except Exception as e:
        console.print(f"[yellow]Could not detect system RAM: {e}. Using defaults.[/yellow]")
        return 16384, 12288


def get_gpu_name(timeout: float = 2.0) -> str | None:
    """Return the GPU name string when available (NVIDIA via nvidia-smi)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout:
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if lines:
                # Return the first GPU name
                return lines[0]
    except Exception:
        pass
    return None


# Small built-in lookup table for typical consumer / datacenter GPUs (approximate GB/s)
_GPU_BANDWIDTH_LOOKUP = {
    # NVIDIA Ampere / Ada examples (approximate)
    "A100": 1555.0,
    "RTX 4090": 1008.0,
    "RTX 3090": 936.0,
    "RTX 3080": 760.0,
    "RTX 3080 Ti": 912.0,
    "RTX 3070": 616.0,
    "RTX 4060": 192.0,
    "GTX 1080": 320.0,
    # AMD examples (approximate)
    "MI100": 1228.0,
    "MI250": 2000.0,
}


def _lookup_bandwidth_by_name(name: str | None) -> float | None:
    if not name:
        return None
    for key, bw in _GPU_BANDWIDTH_LOOKUP.items():
        if key.lower() in name.lower():
            return float(bw)
    return None


def measure_gpu_bandwidth_gbps(sample_bytes: int = 64 * 1024 * 1024) -> float | None:
    """
    Attempt a lightweight GPU memory bandwidth micro-benchmark using Cupy if available.
    Returns GB/s or None if measurement not possible.
    """
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None

    try:
        # allocate a buffer of sample_bytes on device
        n = sample_bytes // 8  # float64
        a = cp.ones(n, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()

        # warm-up
        for _ in range(3):
            b = a.copy()
            cp.cuda.Stream.null.synchronize()

        import time
        runs = 5
        t0 = time.perf_counter()
        for _ in range(runs):
            b = a.copy()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        # bytes copied per run ~ sample_bytes
        bytes_total = sample_bytes * runs
        gbps = (bytes_total / elapsed) / 1e9
        return float(gbps)
    except Exception:
        return None


def get_gpu_bandwidth_gbps(
    gpu_name: str | None = None,
    measured: bool = False,
    force_measure: bool = False,
    available_vram_mb: int | None = None,
    timeout: float = 2.0,
) -> float:
    """
    Return an estimated GPU memory bandwidth in GB/s.

    Logic:
      1. If measured==True, attempt micro-benchmark (may require cupy).
      2. Try lookup by GPU name.
      3. Fallback heuristic based on VRAM size if provided.
      4. Final conservative fallback of 64 GB/s.
    """
    # 1. Attempt measurement
    if measured or force_measure:
        bw = measure_gpu_bandwidth_gbps()
        if bw is not None:
            return bw

    # 2. Try lookup by provided name or detected name
    if gpu_name is None:
        gpu_name = get_gpu_name(timeout=timeout)

    bw_lookup = _lookup_bandwidth_by_name(gpu_name)
    if bw_lookup:
        return bw_lookup

    # 3. Heuristic based on VRAM size
    try:
        if available_vram_mb:
            # rule-of-thumb: ~80 GB/s per GB of HBM/GDDR capacity for rough guess
            # clamp to [20, 2000]
            guess = (available_vram_mb / 1024) * 80
            guess = max(20.0, min(guess, 2000.0))
            return float(guess)
    except Exception:
        pass

    # 4. Conservative fallback
    return 64.0


def estimate_free_vram_bytes(
    mode: str = "auto",
    free_vram_override: int | None = None,
    probes: list | None = None,
    reserved_mb_default: int = 1536,
) -> tuple[int, str, float]:
    """
    Return an estimate of free VRAM in BYTES, not total VRAM.

    Flow:
      - If free_vram_override provided, use it (in MB) and return bytes.
      - Try probe chain (get_gpu_memory()). If the probe returns a total VRAM value
        (MB) we still treat this as a total guess and prefer heuristic when possible.
      - If probe yields a direct free-VRAM probe via `probes` callbacks, use first non-None.
      - Otherwise fall back to a heuristic based on OS and display overheads.

    Returns: (free_bytes, basis, safety_used)
    basis is one of "probe" or "heuristic".
    """
    # If user explicitly provided free_vram override (MB)
    if free_vram_override is not None:
        free_bytes = int(free_vram_override) * 1024 ** 2
        return free_bytes, "probe", 1.0

    # user-supplied probe callbacks take precedence
    if probes:
        for p in probes:
            try:
                val = p()
                if val:
                    # Expect probe to return free MB. If it returns total, caller may have to interpret.
                    free_bytes = int(val) * 1024 ** 2
                    return free_bytes, "probe", 1.0
            except Exception:
                continue

    # Try existing probe chain for total VRAM. If it returns >0 it's a total (not free) value.
    try:
        total_mb = get_gpu_memory()
    except Exception:
        total_mb = 0

    # If probe gave no reliable total, or mode requests heuristic, use heuristic
    if total_mb <= 0 or mode == "heuristic":
        # Heuristic fallback depending on platform
        system = platform.system()
        # detect number of displays (best-effort)
        display_overhead_mb = 0
        try:
            # attempt to detect monitors via screeninfo if available
            import importlib
            screeninfo = importlib.import_module("screeninfo")
            monitors = screeninfo.get_monitors()
            for m in monitors:
                # approximate by width
                w = getattr(m, "width", 0)
                h = getattr(m, "height", 0)
                if w >= 5120 or h >= 2880:
                    display_overhead_mb += 600
                elif w >= 3840 or h >= 2160:
                    display_overhead_mb += 350
                elif w >= 2560 or h >= 1440:
                    display_overhead_mb += 250
                else:
                    display_overhead_mb += 150
        except Exception:
            # If we can't read displays, assume 0 overhead
            display_overhead_mb = 0

        # default heuristics
        if system == "Windows":
            fixed = 1536
            percent = 0.08
            safety = 0.80
        elif system == "Linux":
            # headless detection via DISPLAY env
            if os.environ.get("DISPLAY"):
                fixed = 512
                percent = 0.05
                safety = 0.88
            else:
                fixed = 256
                percent = 0.02
                safety = 0.92
        elif system == "Darwin":
            # Mac: unified memory - be conservative
            fixed = 512
            percent = 0.05
            safety = 0.88
        else:
            fixed = 512
            percent = 0.05
            safety = 0.88

        # If we have a total_mb from probe, use that to compute free guess; otherwise assume 8GB total
        total_mb = total_mb or 8192

        unknown_guard_mb = int(total_mb * 0.10)
        free_guess_mb = int(total_mb - max(fixed, int(total_mb * percent)) - display_overhead_mb - unknown_guard_mb)
        free_bytes = max(0, free_guess_mb) * 1024 ** 2
        return free_bytes, "heuristic", float(safety)

    # If probe returned a total MB > 0, we don't know free exactly — fall back to heuristic path
    # Use heuristic to estimate free from total
    total_mb = total_mb or 8192
    # reuse heuristic path with available total
    # choose same heuristics as above
    system = platform.system()
    if system == "Windows":
        fixed = 1536
        percent = 0.08
        safety = 0.80
    elif system == "Linux":
        if os.environ.get("DISPLAY"):
            fixed = 512
            percent = 0.05
            safety = 0.88
        else:
            fixed = 256
            percent = 0.02
            safety = 0.92
    else:
        fixed = 512
        percent = 0.05
        safety = 0.88

    display_overhead_mb = 0
    unknown_guard_mb = int(total_mb * 0.10)
    free_guess_mb = int(total_mb - max(fixed, int(total_mb * percent)) - display_overhead_mb - unknown_guard_mb)
    free_bytes = max(0, free_guess_mb) * 1024 ** 2
    return free_bytes, "heuristic", float(safety)