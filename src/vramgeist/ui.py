import os
import json
from typing import Dict, Any, Optional
from ._rich_fallback import Panel, Table, Layout, Text, box, Align, Console

# Provide typing aliases so static analyzers treat runtime fallbacks as Any
from typing import Any as _Any
LayoutType: _Any = Layout
PanelType: _Any = Panel
TableType: _Any = Table
AlignType: _Any = Align
TextType: _Any = Text

from .calc import (
    calculate_max_context,
    calculate_vram_usage,
    calculate_ram_usage,
    calculate_semantic_throughput_best_context,
)
from .hw import get_gpu_memory, get_system_memory
from .gguf import estimate_model_size_mb, read_gguf_metadata
from .config import VRAMConfig, DEFAULT_CONFIG
from .bench.llama_bench import measure_tokens_per_second, fit_k_from_measurements

console = Console(force_terminal=True, width=120)


def analyze_gguf_file(path: "os.PathLike[str] | str") -> Dict[str, Any]:
    """
    Minimal analysis function for TUI background processing.
    Delegates to the existing analyze_gguf_file API using defaults.
    Returns the same dict structure as the legacy function below but
    only requires a path argument.
    """
    # Import within function body to avoid circular definitions
    return analyze_gguf_file_with_config(str(path), DEFAULT_CONFIG, None, None)


def create_analysis_layout(model_name: str, status: str = "Initializing...") -> _Any:
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


def update_model_info(layout: _Any, model_name: str, available_vram: int, model_size_mb: float, n_layers: int, status: str = "Analyzing...") -> None:
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


def create_recommendation_panel(model_size_mb: float, n_layers: int, best_gpu_layers: int, best_context: int, available_vram: int) -> _Any:
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


def analyze_gguf_file_with_config(
    filepath: str,
    config: VRAMConfig = DEFAULT_CONFIG,
    vram_override: Optional[int] = None,
    ram_override: Optional[int] = None,
    force_detect: bool = False,
    optimize_for: str = "throughput",
    gpu_bandwidth_gbps: Optional[float] = None,
    measure_bandwidth: bool = False,
    measure_tps: bool = False,
    llama_bin: Optional[str] = None,
    bench_contexts: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a GGUF file and return structured results"""
    model_name = os.path.basename(filepath)
    
    # Step 1: Get GPU VRAM
    available_vram = vram_override if vram_override else get_gpu_memory(force=force_detect)
    
    # Step 1.5: Get system RAM 
    if ram_override:
        total_ram = ram_override
        available_ram = int(ram_override * 0.8)  # Assume 80% available
    else:
        total_ram, available_ram = get_system_memory()
    
    # Step 2: Analyze model size
    model_size_mb = estimate_model_size_mb(filepath)
    
    # Step 3: Read metadata
    metadata, warnings = read_gguf_metadata(filepath)
    
    # Extract layer count
    n_layers = 32  # Default estimate
    if metadata:
        for key, value in metadata.items():
            if 'layer' in key.lower() and 'count' in key.lower():
                if isinstance(value, int):
                    n_layers = value
                    break
            elif key == 'llama.block_count':
                n_layers = value
                break
    
    # Calculate recommendations for different GPU layer configurations
    results = []
    best_gpu_layers = 0
    best_context = 0
    
    # If requested, run token/sec benchmark once per model (before per-gpu_layers loop)
    measured_map: Optional[dict] = None
    measured_k: Optional[float] = None
    if measure_tps:
        # measure_tokens_per_second uses python binding first then binary fallback
        try:
            contexts = [1024, 4096, 8192]
            if bench_contexts:
                try:
                    contexts = [int(x.strip()) for x in bench_contexts.split(",") if x.strip()]
                except Exception:
                    contexts = [1024, 4096, 8192]

            measured_map = measure_tokens_per_second(
                filepath,
                contexts=contexts,
                n_predict=128,
                runs=2,
                warmup=1,
                timeout=60.0,
                use_python_binding=True,
                llama_bin=llama_bin,
            )
            if measured_map:
                measured_k = fit_k_from_measurements(measured_map, eps=1.0)
        except Exception:
            measured_map = None
            measured_k = None

    for gpu_layers in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        # If optimizing for throughput prefer semantic-throughput best context selection
        if optimize_for == "throughput":
            sem = calculate_semantic_throughput_best_context(
                model_size_mb=model_size_mb,
                n_layers=n_layers,
                n_gpu_layers=gpu_layers,
                available_vram_mb=available_vram,
                available_ram_mb=available_ram,
                config=config,
                bw_gbps=gpu_bandwidth_gbps,
                measured_bandwidth=measure_bandwidth,
                measured_tps_map=measured_map,
                measured_k=measured_k,
            )
            max_ctx = sem.get("chosen", 0)
            vram_used = sem.get("vram_used_mb", calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config))
            ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
        else:
            max_ctx = calculate_max_context(model_size_mb, n_layers, gpu_layers, available_vram, available_ram, config)
            vram_used = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
            ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
        
        # Track best option
        if max_ctx > best_context:
            best_context = max_ctx
            best_gpu_layers = gpu_layers
        
        if gpu_layers == 0:
            rec = "CPU only"
        elif gpu_layers == n_layers:
            rec = "Full GPU (fastest)"
        elif vram_used > available_vram * 0.95 or ram_used > available_ram * 0.8:
            rec = "May cause OOM"
        else:
            rec = "Good balance"
        
        results.append({
            "gpu_layers": gpu_layers,
            "max_context": max_ctx,
            "vram_usage_mb": round(vram_used, 1),
            "ram_usage_mb": round(ram_used, 1),
            "recommendation": rec
        })
    
    # Calculate final recommendations
    expected_vram = calculate_vram_usage(model_size_mb, n_layers, best_gpu_layers, best_context, config)
    expected_ram = calculate_ram_usage(model_size_mb, n_layers, best_gpu_layers, best_context, config)
    
    # Generate warnings
    warnings = []
    if best_context < 2048:
        warnings.append("Limited memory may result in poor performance. Consider using a smaller quantized model or adding more memory.")
    
    if expected_vram > available_vram * 0.95:
        warnings.append("VRAM usage is very high. May cause GPU OOM errors.")
    
    if expected_ram > available_ram * 0.8:
        warnings.append("RAM usage is high. May cause system slowdowns.")
    
    return {
        "model": {
            "name": model_name,
            "path": filepath,
            "size_mb": round(model_size_mb, 1),
            "layers": n_layers
        },
        "system": {
            "vram_available_mb": available_vram,
            "ram_total_mb": total_ram,
            "ram_available_mb": available_ram
        },
        "config": config.to_dict(),
        "analysis": results,
        "recommendation": {
            "gpu_layers": best_gpu_layers,
            "max_context": best_context,
            "expected_vram_mb": round(expected_vram, 1),
            "expected_ram_mb": round(expected_ram, 1),
            "total_memory_mb": round(expected_vram + expected_ram, 1)
        },
        "warnings": warnings
    }


def process_gguf_file(
    filepath: str,
    config: VRAMConfig = DEFAULT_CONFIG,
    json_output: bool = False,
    vram_override: Optional[int] = None,
    ram_override: Optional[int] = None,
    force_detect: bool = False,
    optimize_for: str = "throughput",
    gpu_bandwidth_gbps: Optional[float] = None,
    measure_bandwidth: bool = False,
    measure_tps: bool = False,
    llama_bin: Optional[str] = None,
    bench_contexts: Optional[str] = None,
) -> None:
    """Process a single GGUF file and display analysis"""
    if json_output:
        # JSON output mode - no Rich UI
        analysis_result = analyze_gguf_file_with_config(filepath, config, vram_override, ram_override)
        print(json.dumps(analysis_result, indent=2))
        return
    
    # Rich UI mode
    
    # Show header
    console.print(Panel(
        Align.center(
            Text("VRAMGEIST", style="bold magenta") + 
            Text(" - GGUF VRAM Calculator", style="bold blue")
        ),
        style="bright_magenta",
        box=box.DOUBLE
    ))
    
    # Progress updates
    console.print("[bold blue]Detecting GPU VRAM...[/bold blue]")
    console.print("[bold blue]Detecting system RAM...[/bold blue]")
    console.print("[bold blue]Reading model file...[/bold blue]")
    console.print("[bold blue]Reading GGUF metadata...[/bold blue]")
    console.print("[bold blue]Calculating optimal settings...[/bold blue]")
    
    # Get analysis results
    analysis_result = analyze_gguf_file_with_config(
        filepath,
        config,
        vram_override,
        ram_override,
        force_detect,
        optimize_for=optimize_for,
        gpu_bandwidth_gbps=gpu_bandwidth_gbps,
        measure_bandwidth=measure_bandwidth,
        measure_tps=measure_tps,
        llama_bin=llama_bin,
        bench_contexts=bench_contexts,
    )
    
    # Display model info
    model = analysis_result["model"]
    system = analysis_result["system"]
    
    model_info_content = (
        f"[bold blue]Model:[/bold blue] [cyan]{model['name']}[/cyan]\n"
        f"[bold blue]VRAM Available:[/bold blue] [green]{system['vram_available_mb']} MB ({system['vram_available_mb']/1024:.1f} GB)[/green]\n"
        f"[bold blue]RAM Total/Available:[/bold blue] [magenta]{system['ram_total_mb']} MB ({system['ram_total_mb']/1024:.1f} GB) / {system['ram_available_mb']} MB ({system['ram_available_mb']/1024:.1f} GB)[/magenta]\n"
        f"[bold blue]Model Size:[/bold blue] [yellow]{model['size_mb']:.0f} MB ({model['size_mb']/1024:.1f} GB)[/yellow]\n"
        f"[bold blue]Layers:[/bold blue] [cyan]{model['layers']}[/cyan]"
    )
    
    console.print(Panel(
        model_info_content,
        title="[bold magenta]Model Analysis[/bold magenta]",
        style="blue",
        box=box.ROUNDED
    ))
    
    # Show results table
    console.print("\n[bold magenta]Memory Analysis Results:[/bold magenta]")
    console.print("=" * 100)
    console.print(f"{'GPU Layers':<12} {'Max Context':<12} {'VRAM Usage':<12} {'RAM Usage':<12} {'Recommendation'}")
    console.print("-" * 100)
    
    for result in analysis_result["analysis"]:
        console.print(f"[cyan]{result['gpu_layers']:<12}[/cyan] [green]{result['max_context']:,<12}[/green] [yellow]{result['vram_usage_mb']:.0f} MB{'':<6}[/yellow] [magenta]{result['ram_usage_mb']:.0f} MB{'':<6}[/magenta] [white]{result['recommendation']}[/white]")
    
    console.print()
    
    # Show recommendation
    rec = analysis_result["recommendation"]
    console.print("[bold bright_green]RECOMMENDED SETTINGS:[/bold bright_green]")
    console.print("=" * 50)
    console.print(f"[bold bright_green]GPU Layers:[/bold bright_green] [cyan]{rec['gpu_layers']}[/cyan]")
    console.print(f"[bold bright_green]Max Context:[/bold bright_green] [yellow]{rec['max_context']:,}[/yellow]")
    console.print(f"[bold bright_green]Expected VRAM:[/bold bright_green] [yellow]{rec['expected_vram_mb']:.0f} MB ({rec['expected_vram_mb']/1024:.1f} GB)[/yellow]")
    console.print(f"[bold bright_green]Expected RAM:[/bold bright_green] [magenta]{rec['expected_ram_mb']:.0f} MB ({rec['expected_ram_mb']/1024:.1f} GB)[/magenta]")
    console.print(f"[bold bright_green]Total Memory:[/bold bright_green] [white]{rec['total_memory_mb']:.0f} MB ({rec['total_memory_mb']/1024:.1f} GB)[/white]")
    
    # Show warnings
    for warning in analysis_result["warnings"]:
        console.print(f"\n[red]WARNING: {warning}[/red]")
    
    console.print("\n" + "-" * 80 + "\n")