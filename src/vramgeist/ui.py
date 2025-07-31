from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.align import Align

from .calc import (
    calculate_max_context,
    calculate_vram_usage,
)


def create_analysis_layout(model_name: str, status: str = "Initializing...") -> Layout:
    """Create the main analysis layout"""
    layout = Layout()

    header = Panel(
        Align.center(
            Text("VRAMGEIST", style="bold magenta")
            + Text(" - GGUF VRAM Calculator", style="bold blue")
        ),
        style="bright_magenta",
        box=box.DOUBLE,
    )

    model_info = Panel(
        f"[bold blue]📁 Analyzing:[/bold blue] [cyan]{model_name}[/cyan]\n"
        f"[bold purple]Status:[/bold purple] [yellow]{status}[/yellow]",
        title="[bold magenta]Model Analysis[/bold magenta]",
        style="blue",
        box=box.ROUNDED,
    )

    results = Panel(
        "[dim]Analysis results will appear here...[/dim]",
        title="[bold magenta]Results[/bold magenta]",
        style="purple",
        box=box.ROUNDED,
    )

    layout.split_column(
        Layout(header, name="header", size=3),
        Layout(model_info, name="model_info", size=4),
        Layout(results, name="results"),
    )

    return layout


def update_model_info(layout: Layout, model_name: str, available_vram: int, model_size_mb: float, n_layers: int, status: str = "Analyzing...") -> None:
    """Update the model info section"""
    model_info_content = (
        f"[bold blue]📁 Model:[/bold blue] [cyan]{model_name}[/cyan]\n"
        f"[bold blue]💾 VRAM Available:[/bold blue] [green]{available_vram} MB ({available_vram/1024:.1f} GB)[/green]\n"
        f"[bold blue]📊 Model Size:[/bold blue] [yellow]{model_size_mb:.0f} MB ({model_size_mb/1024:.1f} GB)[/yellow]\n"
        f"[bold blue]🔢 Layers:[/bold blue] [cyan]{n_layers}[/cyan]\n"
        f"[bold purple]Status:[/bold purple] [yellow]{status}[/yellow]"
    )

    layout["model_info"].update(
        Panel(
            model_info_content,
            title="[bold magenta]Model Analysis[/bold magenta]",
            style="blue",
            box=box.ROUNDED,
        )
    )


def create_results_table(model_size_mb: float, n_layers: int, available_vram: int):
    """Create the results table with calculations"""
    table = Table(
        title="[bold magenta]VRAM Analysis Results[/bold magenta]",
        box=box.ASCII,
        header_style="bold bright_blue",
        title_style="bold magenta",
    )

    table.add_column("GPU Layers", style="cyan", justify="center")
    table.add_column("Max Context", style="green", justify="center")
    table.add_column("VRAM Usage", style="yellow", justify="center")
    table.add_column("Recommendation", style="white", justify="center")

    best_gpu_layers = n_layers
    best_context = 0

    for gpu_layers in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        max_ctx = calculate_max_context(model_size_mb, n_layers, gpu_layers, available_vram)
        vram_used = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx)

        if gpu_layers == 0:
            rec = "[dim]CPU only[/dim]"
        elif gpu_layers == n_layers:
            rec = "[bright_green]Full GPU (fastest)[/bright_green]"
        elif vram_used > available_vram * 0.95:
            rec = "[red]May cause OOM[/red]"
        else:
            rec = "[blue]⚖️  Good balance[/blue]"

        if max_ctx > best_context and vram_used <= available_vram * 0.9:
            best_context = max_ctx
            best_gpu_layers = gpu_layers

        table.add_row(
            f"{gpu_layers}",
            f"{max_ctx:,}",
            f"{vram_used:.0f} MB",
            rec,
        )

    return table, best_gpu_layers, best_context


def create_recommendation_panel(model_size_mb: float, n_layers: int, best_gpu_layers: int, best_context: int, available_vram: int) -> Panel:
    """Create the final recommendation panel"""
    expected_vram = calculate_vram_usage(model_size_mb, n_layers, best_gpu_layers, best_context)

    if best_context < 2048:
        warning = "\n[red]WARNING: Limited VRAM may result in poor performance.\nConsider using a smaller quantized model or adding more VRAM.[/red]"
        panel_style = "red"
    else:
        warning = ""
        panel_style = "green"

    content = (
        f"[bold bright_green]🎯 GPU Layers:[/bold bright_green] [cyan]{best_gpu_layers}[/cyan]\n"
        f"[bold bright_green]📏 Max Context:[/bold bright_green] [yellow]{best_context:,}[/yellow]\n"
        f"[bold bright_green]💾 Expected VRAM:[/bold bright_green] [magenta]{expected_vram:.0f} MB[/magenta]"
        f"{warning}"
    )

    return Panel(
        content,
        title="[bold bright_green]🏆 RECOMMENDED SETTINGS[/bold bright_green]",
        style=panel_style,
        box=box.DOUBLE,
    )