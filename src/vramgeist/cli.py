from pathlib import Path
import os
import sys
import argparse
import json
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.align import Align

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
        nargs="+",
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
    
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration from profile and CLI args
    config = VRAMConfig.from_profile(args.profile)
    config = config.update_from_args(args)
    
    total_files_processed = 0

    for arg in args.paths:
        path = Path(arg)

        if path.is_file() and path.suffix.lower() == ".gguf":
            if not path.exists():
                console.print(f"[red]Error: File '{path}' not found[/red]")
                continue
            process_gguf_file(str(path), config, args.json)
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
                process_gguf_file(str(gguf_file), config, args.json)
                total_files_processed += 1

        else:
            from glob import glob
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
                        process_gguf_file(gguf_file, config, args.json)
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