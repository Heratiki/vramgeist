from pathlib import Path
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional
from glob import glob
from ._rich_fallback import Console, Panel, box

# Import from modular architecture
from .ui import process_gguf_file
from .config import VRAMConfig

console = Console(force_terminal=True, width=120)



def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all CLI options"""
    parser = argparse.ArgumentParser(
        description="VRAMGEIST - GGUF VRAM Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vramgeist model.gguf
  vramgeist /path/to/models/
  vramgeist *.gguf
  vramgeist model.gguf --json
  vramgeist model.gguf --hidden-size 5120 --vram-safety 0.85
        """
    )
    
    parser.add_argument(
        "paths",
        nargs="*",
        help="Path(s) to GGUF file(s), directory, or glob pattern"
    )
    
    # Configuration options
    parser.add_argument(
        "--profile",
        choices=["default", "conservative", "aggressive"],
        default="default",
        help="Predefined parameter profile (default: default)"
    )
    
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size for context calculation (default: 4096)"
    )
    
    parser.add_argument(
        "--bytes-per-element",
        type=int,
        default=2,
        help="Bytes per element (default: 2 for fp16)"
    )
    
    parser.add_argument(
        "--vram-overhead",
        type=int,
        default=500,
        help="VRAM overhead in MB (default: 500)"
    )
    
    parser.add_argument(
        "--ram-overhead",
        type=int,
        default=1000,
        help="RAM overhead in MB (default: 1000)"
    )
    
    parser.add_argument(
        "--vram-safety",
        type=float,
        default=0.9,
        help="VRAM safety margin (default: 0.9)"
    )
    
    parser.add_argument(
        "--ram-safety",
        type=float,
        default=0.8,
        help="RAM safety margin (default: 0.8)"
    )
    
    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of Rich UI"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--vram-mb",
        type=int,
        help="Override GPU VRAM detection with manual value in MB"
    )
    
    parser.add_argument(
        "--ram-mb",
        type=int,
        help="Override RAM detection with manual value in MB"
    )

    parser.add_argument(
        "--force-detect",
        action="store_true",
        help="Force re-running hardware detection probes (bypass any caching)"
    )

    parser.add_argument(
        "--optimize-for",
        choices=["throughput", "latency", "memory", "balanced"],
        default="balanced",
        help="Optimize recommendations for throughput (tokens/sec), latency (time-to-first-token), memory, or balanced (tradeoff). Default: balanced",
    )

    parser.add_argument(
        "--gpu-bandwidth-gbps",
        type=float,
        help="Override GPU memory bandwidth detection in GB/s (optional)",
    )

    parser.add_argument(
        "--measure-bandwidth",
        action="store_true",
        help="Attempt a lightweight GPU memory bandwidth micro-benchmark (may require cupy).",
    )

    parser.add_argument(
        "--measure-tps",
        action="store_true",
        help="Attempt a small llama.cpp/llama-cpp-python inference micro-benchmark to measure tokens/sec.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug diagnostics in recommendation payloads",
    )

    parser.add_argument(
        "--balanced-weight",
        type=float,
        default=0.35,
        help="When optimize-for=balanced, blend weight for context vs throughput (0..1). Higher favors context (default: 0.35)",
    )

    parser.add_argument(
        "--rebench",
        action="store_true",
        help="Force re-running micro-benchmarks even if a cached result exists for this model",
    )

    parser.add_argument(
        "--llama-bin",
        type=str,
        help="Path to llama.cpp binary to use as fallback for TPS measurement",
    )

    parser.add_argument(
        "--bench-contexts",
        type=str,
        default="1024,4096,8192",
        help="Comma-separated context sizes to sample for TPS measurement (default: 1024,4096,8192)",
    )

    parser.add_argument(
        "--validate-settings",
        action="store_true",
        help="Test recommended settings with actual llama.cpp inference to ensure they work",
    )

    parser.add_argument(
        "--validation-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for validation tests (default: 30.0)",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show saved configuration and exit",
    )

    parser.add_argument(
        "--clear-config",
        action="store_true",
        help="Clear saved configuration and exit",
    )
    
    return parser


def _maybe_env_browse_bypass() -> tuple[int, str | None] | None:
    """
    Support non-interactive testing/automation via env vars:
    - VRAMGEIST_BROWSE_AUTOPATH: if set to a valid path, return (0, absolute path string)
    - VRAMGEIST_BROWSE_CANCEL="1": simulate cancel, return (130, None)
    Returns (exit_code, stdout_string_or_None) or None to continue normal flow.
    """
    auto = os.environ.get("VRAMGEIST_BROWSE_AUTOPATH")
    cancel = os.environ.get("VRAMGEIST_BROWSE_CANCEL")
    if cancel == "1":
        # Simulate cancel
        return (130, None)
    if auto:
        p = Path(auto).expanduser().resolve()
        if p.exists():
            return (0, str(p))
        else:
            # Treat nonexistent path as not selectable; fall through to normal flow
            return None
    return None


def _run_interactive_tui(args) -> None:
    """
    Run the full interactive TUI with analysis and validation controls.
    """
    try:
        from .tui.app import run_tui
        from .tui.options import TUIOptions
    except ImportError:
        msg = (
            "Interactive mode requires Textual TUI dependencies.\n"
            "Install with: pip install vramgeist[tui]\n"
            "Alternatively, set VRAMGEIST_BROWSE_AUTOPATH to bypass interactively."
        )
        console.print(f"[red]{msg}[/red]")
        sys.exit(2)

    # Create TUI options from CLI args
    options = TUIOptions(
        optimize_for=args.optimize_for,
        debug=args.debug,
        validate_settings=args.validate_settings,
        llama_bin=args.llama_bin,
        validation_timeout=args.validation_timeout,
    )
    
    # Run the integrated TUI (no paths = file browser + analysis)
    exit_code = run_tui(paths=[], options=options)
    sys.exit(exit_code)


def main() -> int:
    # If no CLI args provided at all, enter interactive mode (with env bypass)
    if len(sys.argv) == 1:
        # allow tests/automation to bypass
        bypass = _maybe_env_browse_bypass()
        if bypass is not None:
            code, out = bypass
            if out is not None:
                sys.stdout.write(out)
                sys.stdout.flush()
            sys.exit(code)
        # Create minimal args object for TUI
        @dataclass
        class MinimalArgs:
            optimize_for: str = "balanced"
            debug: bool = False
            validate_settings: bool = False
            llama_bin: Optional[str] = None
            validation_timeout: float = 30.0
        
        args = MinimalArgs()
        
        # Auto-load saved llama.cpp path if available
        try:
            from .config_persist import get_llama_bin_path
            saved_llama_bin = get_llama_bin_path()
            if saved_llama_bin:
                args.llama_bin = saved_llama_bin
                args.validate_settings = True  # Enable validation if we have llama.cpp
        except ImportError:
            pass
            
        _run_interactive_tui(args)
        return 0

    parser = create_parser()

    args = parser.parse_args()
    
    # Handle config management commands first
    if args.show_config:
        try:
            from .config_persist import get_config_summary
            summary = get_config_summary()
            console.print("[bold blue]VRAMGeist Configuration:[/bold blue]")
            console.print(f"Config file: [cyan]{summary['config_file']}[/cyan]")
            console.print(f"Config exists: [{'green' if summary['config_exists'] else 'yellow'}]{summary['config_exists']}[/{'green' if summary['config_exists'] else 'yellow'}]")
            console.print(f"Has llama.cpp binary: [{'green' if summary['has_llama_bin'] else 'yellow'}]{summary['has_llama_bin']}[/{'green' if summary['has_llama_bin'] else 'yellow'}]")
            if summary['llama_bin_path']:
                console.print(f"llama.cpp path: [cyan]{summary['llama_bin_path']}[/cyan]")
            console.print(f"Validation timeout: [cyan]{summary['validation_timeout']}s[/cyan]")
            return 0
        except ImportError:
            console.print("[red]Configuration persistence not available[/red]")
            return 1
    
    if args.clear_config:
        try:
            from .config_persist import clear_llama_bin_path, get_config_file
            clear_llama_bin_path()
            console.print("[green]Cleared saved configuration[/green]")
            console.print(f"Config file: [dim]{get_config_file()}[/dim]")
            return 0
        except ImportError:
            console.print("[red]Configuration persistence not available[/red]")
            return 1
    
    # JSON mode should bypass TUI completely
    json_mode = bool(getattr(args, "json", False))

    # Create configuration from profile and CLI args
    config = VRAMConfig.from_profile(args.profile)
    config = config.update_from_args(args)
    
    # Auto-load saved llama.cpp path if not explicitly provided
    if not args.llama_bin and (args.validate_settings or args.measure_tps):
        try:
            from .config_persist import get_llama_bin_path
            saved_llama_bin = get_llama_bin_path()
            if saved_llama_bin:
                args.llama_bin = saved_llama_bin
                if not json_mode:
                    console.print(f"[dim]Using saved llama.cpp binary: {saved_llama_bin}[/dim]")
        except ImportError:
            pass
    
    # If user invoked but provided zero paths after options (e.g. just flags), use full TUI
    if not args.paths:
        bypass = _maybe_env_browse_bypass()
        if bypass is not None:
            code, out = bypass
            if out is not None:
                sys.stdout.write(out)
                sys.stdout.flush()
            sys.exit(code)
        _run_interactive_tui(args)
        return 0

    # Use Rich terminal UI for normal analysis processing

    total_files_processed = 0

    for arg in args.paths:
        path = Path(arg)

        if path.is_file() and path.suffix.lower() == ".gguf":
            if not path.exists():
                console.print(f"[red]Error: File '{path}' not found[/red]")
                continue
            process_gguf_file(
                str(path),
                config,
                args.json,
                args.vram_mb,
                args.ram_mb,
                args.force_detect,
                optimize_for=args.optimize_for,
                gpu_bandwidth_gbps=args.gpu_bandwidth_gbps,
                measure_bandwidth=args.measure_bandwidth,
                measure_tps=args.measure_tps,
                llama_bin=args.llama_bin,
                bench_contexts=args.bench_contexts,
                debug=args.debug,
                validate_settings=args.validate_settings,
                validation_timeout=args.validation_timeout,
            )
            total_files_processed += 1

        elif path.is_dir():
            gguf_files = list(path.glob("*.gguf"))
            if not gguf_files:
                console.print(f"[yellow]No GGUF files found in directory: {path}[/yellow]")
                continue

            if not args.json:
                console.print(Panel(
                    f"[bold blue]Found {len(gguf_files)} GGUF files in:[/bold blue] [cyan]{path}[/cyan]",
                    style="blue",
                    box=box.ROUNDED
                ))

                for gguf_file in sorted(gguf_files):
                    process_gguf_file(
                        str(gguf_file),
                        config,
                        args.json,
                        args.vram_mb,
                        args.ram_mb,
                        args.force_detect,
                        optimize_for=args.optimize_for,
                        gpu_bandwidth_gbps=args.gpu_bandwidth_gbps,
                        measure_bandwidth=args.measure_bandwidth,
                        measure_tps=args.measure_tps,
                        llama_bin=args.llama_bin,
                        bench_contexts=args.bench_contexts,
                        debug=args.debug,
                        validate_settings=args.validate_settings,
                        validation_timeout=args.validation_timeout,
                    )
                total_files_processed += 1

        else:
            matches = glob(str(path))
            gguf_matches = [f for f in matches if f.lower().endswith(".gguf")]

            if gguf_matches:
                if len(gguf_matches) > 1 and not args.json:
                    console.print(Panel(
                        f"[bold blue]Found {len(gguf_matches)} GGUF files matching pattern:[/bold blue] [cyan]{path}[/cyan]",
                        style="blue",
                        box=box.ROUNDED
                    ))

                for gguf_file in sorted(gguf_matches):
                    if os.path.exists(gguf_file):
                        process_gguf_file(
                            gguf_file,
                            config,
                            args.json,
                            args.vram_mb,
                            args.ram_mb,
                            args.force_detect,
                            optimize_for=args.optimize_for,
                            gpu_bandwidth_gbps=args.gpu_bandwidth_gbps,
                            measure_bandwidth=args.measure_bandwidth,
                            measure_tps=args.measure_tps,
                            llama_bin=args.llama_bin,
                            bench_contexts=args.bench_contexts,
                            debug=args.debug,
                            validate_settings=args.validate_settings,
                            validation_timeout=args.validation_timeout,
                        )
                        total_files_processed += 1
            else:
                console.print(f"[red]Error: '{path}' is not a valid GGUF file, directory, or pattern[/red]")

    if total_files_processed > 0 and not args.json:
        console.print(Panel(
            f"[bold green]Analysis complete![/bold green]\n"
            f"[bold blue]Total files processed:[/bold blue] [cyan]{total_files_processed}[/cyan]",
            title="[bold bright_green]Summary[/bold bright_green]",
            style="bright_green",
            box=box.DOUBLE
        ))

    # Return success exit code
    return 0


# Top-level guard for proper exit code and output handling
if __name__ == "__main__":
    sys.exit(main())