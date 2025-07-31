from pathlib import Path
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.align import Align

# Temporary imports from monolith until functions are extracted to modules
# After extraction, replace with: from .hw import get_gpu_memory, get_system_memory, etc.
from vramgeist import (
    get_gpu_memory,
    get_system_memory,
    estimate_model_size_mb,
    read_gguf_metadata,
    calculate_max_context,
    calculate_vram_usage,
    calculate_ram_usage,
)

console = Console(force_terminal=True, width=120)


def process_gguf_file(filepath: str) -> None:
    """Shim: delegate to existing vramgeist.process_gguf_file during transition."""
    # Import locally to avoid circular import once we move functionality out
    from vramgeist import process_gguf_file as _process
    _process(filepath)


def _print_usage_and_exit() -> None:
    console.print(Panel(
        "[bold blue]Usage:[/bold blue] [cyan]vramgeist <path_to_gguf_file_or_folder>[/cyan]\n\n"
        "[bold magenta]Examples:[/bold magenta]\n"
        "[dim]-[/dim] [green]vramgeist model.gguf[/green]\n"
        "[dim]-[/dim] [green]vramgeist /path/to/models/[/green]\n"
        "[dim]-[/dim] [green]vramgeist *.gguf[/green]",
        title="[bold bright_magenta]VRAMGEIST - GGUF VRAM Calculator[/bold bright_magenta]",
        style="bright_blue",
        box=box.DOUBLE
    ))
    sys.exit(1)


def main() -> None:
    # Maintain current CLI UX exactly
    if len(sys.argv) < 2:
        _print_usage_and_exit()

    total_files_processed = 0

    for arg in sys.argv[1:]:
        path = Path(arg)

        if path.is_file() and path.suffix.lower() == ".gguf":
            if not path.exists():
                console.print(f"[red]❌ Error: File '{path}' not found[/red]")
                continue
            process_gguf_file(str(path))
            total_files_processed += 1

        elif path.is_dir():
            gguf_files = list(path.glob("*.gguf"))
            if not gguf_files:
                console.print(f"[yellow]No GGUF files found in directory: {path}[/yellow]")
                continue

            console.print(Panel(
                f"[bold blue]📂 Found {len(gguf_files)} GGUF files in:[/bold blue] [cyan]{path}[/cyan]",
                style="blue",
                box=box.ROUNDED
            ))

            for gguf_file in sorted(gguf_files):
                process_gguf_file(str(gguf_file))
                total_files_processed += 1

        else:
            from glob import glob
            matches = glob(str(path))
            gguf_matches = [f for f in matches if f.lower().endswith(".gguf")]

            if gguf_matches:
                if len(gguf_matches) > 1:
                    console.print(Panel(
                        f"[bold blue]🔍 Found {len(gguf_matches)} GGUF files matching pattern:[/bold blue] [cyan]{path}[/cyan]",
                        style="blue",
                        box=box.ROUNDED
                    ))

                for gguf_file in sorted(gguf_matches):
                    if os.path.exists(gguf_file):
                        process_gguf_file(gguf_file)
                        total_files_processed += 1
            else:
                console.print(f"[red]❌ Error: '{path}' is not a valid GGUF file, directory, or pattern[/red]")

    if total_files_processed > 0:
        console.print(Panel(
            f"[bold green]Analysis complete![/bold green]\n"
            f"[bold blue]Total files processed:[/bold blue] [cyan]{total_files_processed}[/cyan]",
            title="[bold bright_green]Summary[/bold bright_green]",
            style="bright_green",
            box=box.DOUBLE
        ))