#!/usr/bin/env python3
"""
GGUF VRAM Calculator - Calculate max context size for GGUF models
Drag and drop a GGUF model file to calculate optimal context size based on available VRAM
"""

import sys
import os
import struct
import subprocess
import platform
import time
import psutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.align import Align

console = Console(force_terminal=True, width=120)

def get_gpu_memory():
    """Get available GPU VRAM in MB"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            console.print("[yellow]Could not detect GPU VRAM. Using default 8GB.[/yellow]")
            return 8192
    except FileNotFoundError:
        console.print("[yellow]nvidia-smi not found. Using default 8GB VRAM.[/yellow]")
        return 8192

def get_system_memory():
    """Get available system RAM in MB"""
    try:
        # Get system memory info using psutil
        memory = psutil.virtual_memory()
        total_ram_mb = memory.total // (1024 * 1024)
        available_ram_mb = memory.available // (1024 * 1024)
        return total_ram_mb, available_ram_mb
    except Exception as e:
        console.print(f"[yellow]Could not detect system RAM: {e}. Using defaults.[/yellow]")
        return 16384, 12288  # Default: 16GB total, 12GB available

def read_gguf_metadata(filepath):
    """Extract basic metadata from GGUF file"""
    try:
        with open(filepath, 'rb') as f:
            # Read GGUF header
            magic = f.read(4)
            if magic != b'GGUF':
                return None
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            # Sanity check for metadata count
            if metadata_kv_count > 10000:  # Reasonable limit
                metadata_kv_count = 100
            
            # Parse metadata
            metadata = {}
            for _ in range(metadata_kv_count):
                # Read key length and key
                key_len = struct.unpack('<Q', f.read(8))[0]
                
                # Sanity check for key length
                if key_len > 1024:  # Keys shouldn't be longer than 1KB
                    break
                    
                key = f.read(key_len).decode('utf-8')
                
                # Read value type
                value_type = struct.unpack('<I', f.read(4))[0]
                
                # Read value based on type
                if value_type == 4:  # STRING
                    value_len = struct.unpack('<Q', f.read(8))[0]
                    value = f.read(value_len).decode('utf-8')
                elif value_type == 5:  # ARRAY
                    array_type = struct.unpack('<I', f.read(4))[0]
                    array_len = struct.unpack('<Q', f.read(8))[0]
                    # Skip array data for now
                    if array_type == 4:  # STRING array
                        for _ in range(array_len):
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            f.read(str_len)
                    value = f"[array of {array_len} items]"
                elif value_type in [6, 7]:  # INT32, UINT32
                    value = struct.unpack('<I', f.read(4))[0]
                elif value_type in [8, 9]:  # INT64, UINT64
                    value = struct.unpack('<Q', f.read(8))[0]
                elif value_type in [10, 11]:  # FLOAT32, FLOAT64
                    if value_type == 10:
                        value = struct.unpack('<f', f.read(4))[0]
                    else:
                        value = struct.unpack('<d', f.read(8))[0]
                elif value_type == 12:  # BOOL
                    value = struct.unpack('<?', f.read(1))[0]
                else:
                    # Skip unknown types
                    continue
                
                metadata[key] = value
            
            return metadata
    except Exception as e:
        console.print(f"[red]Error reading GGUF metadata: {e}[/red]")
        return None

def estimate_model_size_mb(filepath):
    """Estimate model size in MB"""
    try:
        file_size = os.path.getsize(filepath)
        return file_size / (1024 * 1024)
    except:
        return 0

def calculate_vram_usage(model_size_mb, n_layers, n_gpu_layers, context_length):
    """
    Calculate VRAM usage based on oobabooga's formula approximation
    
    VRAM = Model_VRAM + Context_VRAM + Overhead
    
    Model_VRAM = (model_size_mb * gpu_layers_ratio)
    Context_VRAM = context_length * hidden_size * 2 * bytes_per_element / (1024*1024)
    Overhead = ~500MB for llama.cpp operations
    """
    
    # Calculate what fraction of layers are on GPU
    gpu_layers_ratio = min(n_gpu_layers / n_layers, 1.0) if n_layers > 0 else 0.8
    
    # Model VRAM usage (layers on GPU)
    model_vram = model_size_mb * gpu_layers_ratio
    
    # Context cache VRAM (approximate formula)
    # For typical models: hidden_size ~= 4096, 2 bytes per element (fp16)
    # This is a simplified approximation
    hidden_size = 4096  # Approximate for most 7B-13B models
    bytes_per_element = 2  # fp16
    context_vram = (context_length * hidden_size * 2 * bytes_per_element) / (1024 * 1024)
    
    # llama.cpp overhead
    overhead = 500
    
    total_vram = model_vram + context_vram + overhead
    return total_vram

def calculate_ram_usage(model_size_mb, n_layers, n_gpu_layers, context_length):
    """
    Calculate RAM usage for CPU layers and system overhead
    
    RAM = Model_RAM + Context_RAM + System_Overhead
    
    Model_RAM = (model_size_mb * cpu_layers_ratio)
    Context_RAM = context_length * hidden_size * 2 * bytes_per_element / (1024*1024)  
    System_Overhead = ~1GB for llama.cpp + OS overhead
    """
    
    # Calculate what fraction of layers are on CPU
    cpu_layers = n_layers - n_gpu_layers
    cpu_layers_ratio = cpu_layers / n_layers if n_layers > 0 else 0.2
    
    # Model RAM usage (layers on CPU)
    model_ram = model_size_mb * cpu_layers_ratio
    
    # Context cache RAM (shared with CPU processing)
    hidden_size = 4096
    bytes_per_element = 2  # fp16
    context_ram = (context_length * hidden_size * 2 * bytes_per_element) / (1024 * 1024)
    
    # System overhead (llama.cpp + OS)
    system_overhead = 1024  # 1GB
    
    total_ram = model_ram + context_ram + system_overhead
    return total_ram

def calculate_total_memory_usage(model_size_mb, n_layers, n_gpu_layers, context_length):
    """Calculate combined VRAM + RAM usage"""
    vram_usage = calculate_vram_usage(model_size_mb, n_layers, n_gpu_layers, context_length)
    ram_usage = calculate_ram_usage(model_size_mb, n_layers, n_gpu_layers, context_length)
    return vram_usage, ram_usage

def calculate_max_context(model_size_mb, n_layers, n_gpu_layers, available_vram_mb, available_ram_mb=None):
    """Calculate maximum context length for given VRAM and RAM constraints"""
    
    # Reserve safety margins
    vram_safety_margin = 0.9
    ram_safety_margin = 0.8  # More conservative for RAM due to OS needs
    
    usable_vram = available_vram_mb * vram_safety_margin
    
    gpu_layers_ratio = min(n_gpu_layers / n_layers, 1.0) if n_layers > 0 else 0.8
    model_vram = model_size_mb * gpu_layers_ratio
    vram_overhead = 500
    
    # Available VRAM for context
    context_vram_budget = usable_vram - model_vram - vram_overhead
    
    if context_vram_budget <= 0:
        return 0
    
    # Calculate VRAM-limited context
    hidden_size = 4096
    bytes_per_element = 2
    
    vram_max_context = int((context_vram_budget * 1024 * 1024) / (hidden_size * 2 * bytes_per_element))
    
    # If RAM info is available, also check RAM constraints
    if available_ram_mb is not None:
        usable_ram = available_ram_mb * ram_safety_margin
        
        cpu_layers_ratio = (n_layers - n_gpu_layers) / n_layers if n_layers > 0 else 0.2
        model_ram = model_size_mb * cpu_layers_ratio
        ram_overhead = 1024  # 1GB system overhead
        
        context_ram_budget = usable_ram - model_ram - ram_overhead
        
        if context_ram_budget > 0:
            ram_max_context = int((context_ram_budget * 1024 * 1024) / (hidden_size * 2 * bytes_per_element))
            # Use the more restrictive constraint
            max_context = min(vram_max_context, ram_max_context)
        else:
            max_context = vram_max_context
    else:
        max_context = vram_max_context
    
    # Round down to nearest power of 2 or common context sizes
    # Extended for large context models and high-memory systems
    common_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 
                   262144, 524288, 1048576, 2097152, 4194304, 8388608]
    for size in reversed(common_sizes):
        if max_context >= size:
            return size
    
    return 512  # Minimum reasonable context

def create_analysis_layout(model_name, status="Initializing..."):
    """Create the main analysis layout"""
    layout = Layout()
    
    # Create header
    header = Panel(
        Align.center(
            Text("VRAMGEIST", style="bold magenta") + 
            Text(" - GGUF VRAM Calculator", style="bold blue")
        ),
        style="bright_magenta",
        box=box.DOUBLE
    )
    
    # Create model info panel
    model_info = Panel(
        f"[bold blue]📁 Analyzing:[/bold blue] [cyan]{model_name}[/cyan]\n"
        f"[bold purple]Status:[/bold purple] [yellow]{status}[/yellow]",
        title="[bold magenta]Model Analysis[/bold magenta]",
        style="blue",
        box=box.ROUNDED
    )
    
    # Create empty results panel initially
    results = Panel(
        "[dim]Analysis results will appear here...[/dim]",
        title="[bold magenta]Results[/bold magenta]",
        style="purple",
        box=box.ROUNDED
    )
    
    layout.split_column(
        Layout(header, name="header", size=3),
        Layout(model_info, name="model_info", size=4),
        Layout(results, name="results")
    )
    
    return layout

def update_model_info(layout, model_name, available_vram, model_size_mb, n_layers, status="Analyzing..."):
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
            box=box.ROUNDED
        )
    )

def create_results_table(model_size_mb, n_layers, available_vram):
    """Create the results table with calculations"""
    table = Table(
        title="[bold magenta]VRAM Analysis Results[/bold magenta]",
        box=box.ASCII,
        header_style="bold bright_blue",
        title_style="bold magenta"
    )
    
    table.add_column("GPU Layers", style="cyan", justify="center")
    table.add_column("Max Context", style="green", justify="center")  
    table.add_column("VRAM Usage", style="yellow", justify="center")
    table.add_column("Recommendation", style="white", justify="center")
    
    # Calculate for different GPU layer settings
    best_gpu_layers = n_layers
    best_context = 0
    
    for gpu_layers in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        max_ctx = calculate_max_context(model_size_mb, n_layers, gpu_layers, available_vram)
        vram_used = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx)
        
        if gpu_layers == 0:
            rec = "[dim]CPU only[/dim]"
            rec_style = "dim"
        elif gpu_layers == n_layers:
            rec = "[bright_green]Full GPU (fastest)[/bright_green]"
            rec_style = "bright_green"
        elif vram_used > available_vram * 0.95:
            rec = "[red]May cause OOM[/red]"
            rec_style = "red"
        else:
            rec = "[blue]⚖️  Good balance[/blue]"
            rec_style = "blue"
        
        # Track best option for recommendation
        if max_ctx > best_context and vram_used <= available_vram * 0.9:
            best_context = max_ctx
            best_gpu_layers = gpu_layers
        
        table.add_row(
            f"{gpu_layers}",
            f"{max_ctx:,}",
            f"{vram_used:.0f} MB",
            rec
        )
    
    return table, best_gpu_layers, best_context

def create_recommendation_panel(model_size_mb, n_layers, best_gpu_layers, best_context, available_vram):
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
        box=box.DOUBLE
    )

def process_gguf_file(filepath):
    """Process a single GGUF file and display analysis with Rich UI"""
    model_name = os.path.basename(filepath)
    
    # Show header
    console.print(Panel(
        Align.center(
            Text("VRAMGEIST", style="bold magenta") + 
            Text(" - GGUF VRAM Calculator", style="bold blue")
        ),
        style="bright_magenta",
        box=box.DOUBLE
    ))
    
    # Step 1: Show initial info and get GPU VRAM
    console.print("[bold blue]Detecting GPU VRAM...[/bold blue]")
    available_vram = get_gpu_memory()
    
    # Step 1.5: Get system RAM 
    console.print("[bold blue]Detecting system RAM...[/bold blue]")
    total_ram, available_ram = get_system_memory()
    
    # Step 2: Analyze model size
    console.print("[bold blue]Reading model file...[/bold blue]")
    model_size_mb = estimate_model_size_mb(filepath)
    
    # Step 3: Read metadata
    console.print("[bold blue]Reading GGUF metadata...[/bold blue]")
    metadata = read_gguf_metadata(filepath)
    
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
    
    # Show model info
    model_info_content = (
        f"[bold blue]Model:[/bold blue] [cyan]{model_name}[/cyan]\n"
        f"[bold blue]VRAM Available:[/bold blue] [green]{available_vram} MB ({available_vram/1024:.1f} GB)[/green]\n"
        f"[bold blue]RAM Total/Available:[/bold blue] [magenta]{total_ram} MB ({total_ram/1024:.1f} GB) / {available_ram} MB ({available_ram/1024:.1f} GB)[/magenta]\n"
        f"[bold blue]Model Size:[/bold blue] [yellow]{model_size_mb:.0f} MB ({model_size_mb/1024:.1f} GB)[/yellow]\n"
        f"[bold blue]Layers:[/bold blue] [cyan]{n_layers}[/cyan]"
    )
    
    console.print(Panel(
        model_info_content,
        title="[bold magenta]Model Analysis[/bold magenta]",
        style="blue",
        box=box.ROUNDED
    ))
    
    # Step 4: Calculate recommendations
    console.print("[bold blue]Calculating optimal settings...[/bold blue]")
    
    # Show results as simple text to avoid Unicode issues
    console.print("\n[bold magenta]Memory Analysis Results:[/bold magenta]")
    console.print("=" * 100)
    console.print(f"{'GPU Layers':<12} {'Max Context':<12} {'VRAM Usage':<12} {'RAM Usage':<12} {'Recommendation'}")
    console.print("-" * 100)
    
    best_gpu_layers = 0
    best_context = 0
    
    for gpu_layers in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        max_ctx = calculate_max_context(model_size_mb, n_layers, gpu_layers, available_vram, available_ram)
        vram_used = calculate_vram_usage(model_size_mb, n_layers, gpu_layers, max_ctx)
        ram_used = calculate_ram_usage(model_size_mb, n_layers, gpu_layers, max_ctx)
        
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
        
        console.print(f"[cyan]{gpu_layers:<12}[/cyan] [green]{max_ctx:,<12}[/green] [yellow]{vram_used:.0f} MB{'':<6}[/yellow] [magenta]{ram_used:.0f} MB{'':<6}[/magenta] [white]{rec}[/white]")
    
    console.print()
    
    # Show recommendation as simple text
    expected_vram = calculate_vram_usage(model_size_mb, n_layers, best_gpu_layers, best_context)
    expected_ram = calculate_ram_usage(model_size_mb, n_layers, best_gpu_layers, best_context)
    
    console.print("[bold bright_green]RECOMMENDED SETTINGS:[/bold bright_green]")
    console.print("=" * 50)
    console.print(f"[bold bright_green]GPU Layers:[/bold bright_green] [cyan]{best_gpu_layers}[/cyan]")
    console.print(f"[bold bright_green]Max Context:[/bold bright_green] [yellow]{best_context:,}[/yellow]")
    console.print(f"[bold bright_green]Expected VRAM:[/bold bright_green] [yellow]{expected_vram:.0f} MB ({expected_vram/1024:.1f} GB)[/yellow]")
    console.print(f"[bold bright_green]Expected RAM:[/bold bright_green] [magenta]{expected_ram:.0f} MB ({expected_ram/1024:.1f} GB)[/magenta]")
    console.print(f"[bold bright_green]Total Memory:[/bold bright_green] [white]{expected_vram + expected_ram:.0f} MB ({(expected_vram + expected_ram)/1024:.1f} GB)[/white]")
    
    # Enhanced warnings
    if best_context < 2048:
        console.print("\n[red]WARNING: Limited memory may result in poor performance.[/red]")
        console.print("[red]Consider using a smaller quantized model or adding more memory.[/red]")
    
    if expected_vram > available_vram * 0.95:
        console.print("\n[red]WARNING: VRAM usage is very high. May cause GPU OOM errors.[/red]")
    
    if expected_ram > available_ram * 0.8:
        console.print("\n[red]WARNING: RAM usage is high. May cause system slowdowns.[/red]")
    
    console.print("\n" + "-" * 80 + "\n")

def main():
    if len(sys.argv) < 2:
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
    
    # Initial banner removed - each file shows its own header
    
    # Process all arguments as potential paths
    total_files_processed = 0
    
    for arg in sys.argv[1:]:
        path = Path(arg)
        
        if path.is_file() and path.suffix.lower() == '.gguf':
            if not path.exists():
                console.print(f"[red]❌ Error: File '{path}' not found[/red]")
                continue
            process_gguf_file(str(path))
            total_files_processed += 1
            
        elif path.is_dir():
            # Process all GGUF files in directory
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
            # Try glob pattern matching
            from glob import glob
            matches = glob(str(path))
            gguf_matches = [f for f in matches if f.lower().endswith('.gguf')]
            
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
    
    # Final summary
    if total_files_processed > 0:
        console.print(Panel(
            f"[bold green]Analysis complete![/bold green]\n"
            f"[bold blue]Total files processed:[/bold blue] [cyan]{total_files_processed}[/cyan]",
            title="[bold bright_green]Summary[/bold bright_green]",
            style="bright_green",
            box=box.DOUBLE
        ))

if __name__ == "__main__":
    main()