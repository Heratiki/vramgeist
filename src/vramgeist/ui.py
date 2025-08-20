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
from .validate import validate_recommendation
from .hw import get_gpu_memory, get_system_memory
from .gguf import estimate_model_size_mb, read_gguf_metadata
from .config import VRAMConfig, DEFAULT_CONFIG
from .bench.llama_bench import measure_tokens_per_second, fit_k_from_measurements
import hashlib
import pathlib
import pickle

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

    # prefer GPU when contexts tie; initialize to sentinel values
    best_gpu_layers = None
    best_context = -1

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

        # prefer higher GPU layer counts when max context ties
        if vram_used <= available_vram * 0.9 and (
            max_ctx > best_context or (max_ctx == best_context and gpu_layers > (best_gpu_layers or -1))
        ):
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
    debug: bool = False,
    balanced_weight: float = 0.35,
    rebench: bool = False,
    validate_settings: bool = False,
    validation_timeout: float = 30.0,
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
    # Track best candidate. If throughput optimization is used, prefer higher score.
    best_gpu_layers = 0
    best_context = 0
    best_score = -1.0
    
    # If requested, run token/sec benchmark once per model (before per-gpu_layers loop)
    measured_map: Optional[dict] = None
    measured_k: Optional[float] = None
    bench_contexts_list: Optional[list[int]] = None
    if measure_tps:
        # measure_tokens_per_second uses python binding first then binary fallback
        try:
            contexts = [1024, 4096, 8192]
            if bench_contexts:
                try:
                    contexts = [int(x.strip()) for x in bench_contexts.split(",") if x.strip()]
                except Exception:
                    contexts = [1024, 4096, 8192]

            bench_contexts_list = contexts

            # Persistent bench cache keyed by model file hash + bench contexts
            try:
                h = hashlib.sha256()
                with open(filepath, "rb") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
                model_hash = h.hexdigest()
            except Exception:
                model_hash = None

            cache_dir = pathlib.Path(os.path.expanduser("~")) / ".cache" / "vramgeist" / "bench"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{model_hash}.pkl" if model_hash else None

            if cache_path and cache_path.exists() and not rebench:
                try:
                    with open(cache_path, "rb") as cf:
                        cached = pickle.load(cf)
                        measured_map = cached.get("measured_map")
                        measured_k = cached.get("measured_k")
                except Exception:
                    measured_map = None
                    measured_k = None

            if measured_map is None:
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
                # Bench helper may return a rich structure {"map": {...}, "details": {...}}
                if measured_map and isinstance(measured_map, dict) and "map" in measured_map:
                    measured_k = fit_k_from_measurements(measured_map, eps=1.0)
                    # extract the simple map for downstream calculations
                    measured_map = measured_map.get("map", {})
                else:
                    if measured_map:
                        measured_k = fit_k_from_measurements(measured_map, eps=1.0)

                # persist
                if cache_path:
                    try:
                        with open(cache_path, "wb") as cf:
                            pickle.dump({"measured_map": measured_map, "measured_k": measured_k}, cf)
                    except Exception:
                        pass
        except Exception:
            measured_map = None
            measured_k = None

    for gpu_layers in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        sem = {}
        # Two modes: throughput (semantic), memory (pure memory-bound), or balanced (tradeoff)
        # build a small model_meta from metadata when available
        model_meta = {}
        for k in ("n_kv_heads", "head_dim", "kv_dtype", "weight_bytes_vram", "n_heads", "hidden_size"):
            if metadata and k in metadata:
                model_meta[k] = metadata[k]
        model_meta["n_layers"] = n_layers

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
                model_meta=model_meta,
                opts={"debug": debug},
            )
            max_ctx = sem.get("chosen", 0)
            # prefer vram_used declared in semantic result (when model_meta path used)
            vram_used = sem.get("vram_used_mb") if sem and isinstance(sem, dict) and sem.get("vram_used_mb") is not None else calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
            ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)

        elif optimize_for == "memory":
            # Pure memory-bound selection: maximize context that fits memory
            max_ctx = calculate_max_context(model_size_mb, n_layers, gpu_layers, available_vram, available_ram, config)
            vram_used = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
            ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
            # keep sem minimal so scoring code uses fallback path
            sem = {"chosen": max_ctx, "vram_used_mb": vram_used}

        elif optimize_for == "balanced":
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
                model_meta=model_meta,
                opts={"debug": debug},
            )

            # Recompute max_ctx from semantic result but nudge toward larger contexts when memory margin allows.
            max_ctx = sem.get("chosen", 0)
            vram_used = sem.get("vram_used_mb") if sem and isinstance(sem, dict) and sem.get("vram_used_mb") is not None else calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
            ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)

            # If measured_map exists we will rescore candidates using a blended score
            # But only if we have sufficient measured data points (at least 3)
            if measured_map and len(measured_map) >= 3:
                # Build candidate list (recompute common sizes to check)
                common_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
                # compute normalized TPS and context values
                candidates = []
                for C in common_sizes:
                    vram_used_c = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, C, config)
                    if vram_used_c > available_vram * config.vram_safety_margin:
                        continue
                    # get measured tps when present, else skip
                    tps_c = None
                    if measured_map and C in measured_map:
                        tps_c = float(measured_map[C])
                    candidates.append((C, vram_used_c, tps_c))

                # compute score normalization ranges
                tps_values = [c[2] for c in candidates if c[2] is not None]
                ctx_values = [c[0] for c in candidates]
                if tps_values and ctx_values:
                    tmin, tmax = min(tps_values), max(tps_values)
                    cmin, cmax = min(ctx_values), max(ctx_values)
                    best_local = None
                    best_local_score = -1.0
                    for C, vram_c, tps_c in candidates:
                        # normalize
                        tnorm = (tps_c - tmin) / (tmax - tmin) if (tmax - tmin) > 0 and tps_c is not None else 0.0
                        cnorm = (C - cmin) / (cmax - cmin) if (cmax - cmin) > 0 else 0.0
                        # blended score — balanced_weight favors context
                        alpha = float(balanced_weight)
                        blended = (1.0 - alpha) * tnorm + alpha * cnorm
                        if blended > best_local_score:
                            best_local_score = blended
                            best_local = (C, vram_c)
                    if best_local:
                        max_ctx = best_local[0]
                        vram_used = best_local[1]
                        ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)

            # If there's plenty of VRAM margin, try bumping context up one step (choose next common size) to favor usefulness.
            common_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
            try:
                idx = common_sizes.index(max_ctx) if max_ctx in common_sizes else None
            except Exception:
                idx = None
            if idx is not None and idx + 1 < len(common_sizes):
                candidate = common_sizes[idx + 1]
                candidate_vram = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, candidate, config)
                # allow bump if VRAM stays under 92% of usable_vram
                usable_vram_mb = available_vram * config.vram_safety_margin
                if candidate_vram <= usable_vram_mb * 0.92:
                    max_ctx = candidate
                    vram_used = candidate_vram
                    ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)

        else:
            # fallback to previous behavior
            max_ctx = calculate_max_context(model_size_mb, n_layers, gpu_layers, available_vram, available_ram, config)
            vram_used = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
            ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx, config)
        
        # Track best option
        # If the semantic throughput dict was used, it may include a 'score' (higher is better)
        candidate_score = sem.get("score") if optimize_for == "throughput" and isinstance(sem, dict) else None

        if candidate_score is not None:
            # Prefer higher semantic score; tie-break by higher context then more GPU layers
            if candidate_score > best_score or (
                candidate_score == best_score and (max_ctx > best_context or (max_ctx == best_context and gpu_layers > best_gpu_layers))
            ):
                best_score = candidate_score
                best_context = max_ctx
                best_gpu_layers = gpu_layers
        else:
            # Default: prefer larger context; tie-break by more GPU layers
            if max_ctx > best_context or (max_ctx == best_context and gpu_layers > best_gpu_layers):
                best_context = max_ctx
                best_gpu_layers = gpu_layers

        # ensure numeric values for comparisons/formatting
        vram_used = 0 if vram_used is None else vram_used
        ram_used = 0 if ram_used is None else ram_used

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
            "vram_usage_mb": round(float(vram_used), 1),
            "ram_usage_mb": round(float(ram_used), 1),
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
    
    # Run validation if requested
    validation_result = None
    if validate_settings and llama_bin:
        validation_result = validate_recommendation(
            model_path=filepath,
            recommended_gpu_layers=best_gpu_layers,
            recommended_context=best_context,
            llama_bin=llama_bin
        )
        
        # Add validation warnings to main warnings
        if validation_result and not validation_result.get("validated", False):
            warnings.append(f"⚠️  Validation failed: {validation_result.get('reason', 'Unknown error')}")
            for rec in validation_result.get("recommendations", []):
                warnings.append(rec)
    elif validate_settings and not llama_bin:
        warnings.append("⚠️  Validation requested but no llama.cpp binary provided (use --llama-bin)")
    
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
        "benchmarks": {
            "measured_map": measured_map,
            "fitted_k": measured_k,
            "contexts": bench_contexts_list,
        },
        "recommendation": {
            "gpu_layers": best_gpu_layers,
            "max_context": best_context,
            "expected_vram_mb": round(expected_vram, 1),
            "expected_ram_mb": round(expected_ram, 1),
            "total_memory_mb": round(expected_vram + expected_ram, 1)
        },
        "validation": validation_result,
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
    debug: bool = False,
    validate_settings: bool = False,
    validation_timeout: float = 30.0,
) -> None:
    """Process a single GGUF file and display analysis"""
    if json_output:
        # JSON output mode - no Rich UI
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
            debug=debug,
            validate_settings=validate_settings,
            validation_timeout=validation_timeout,
        )
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
        debug=debug,
        validate_settings=validate_settings,
        validation_timeout=validation_timeout,
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