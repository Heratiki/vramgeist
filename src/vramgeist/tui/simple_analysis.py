"""
Simple, stable Analysis TUI for GGUF files with validation controls.

This replaces the complex app.py implementation with a cleaner, more reliable approach.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Vertical

from .options import TUIOptions


class SimpleAnalysisApp(App):
    """Simple, stable analysis TUI that shows results for GGUF files."""
    
    CSS = """
    .settings {
        background: blue;
        color: white;
        text-style: bold;
        text-align: center;
        margin: 1;
        padding: 1;
    }
    #results-container {
        overflow: auto;
        height: 1fr;
    }
    .results {
        margin: 1;
        padding: 1;
    }
    .analyzing {
        color: yellow;
        text-style: bold;
    }
    .completed {
        color: green;
        text-style: bold;
    }
    .error {
        color: red;
        text-style: bold;
    }
    """
    
    def __init__(self, paths: List[Path], options: TUIOptions):
        super().__init__()
        self.paths = paths
        self.options = options
        self.results: Dict[Path, Dict[str, Any]] = {}
        self.errors: Dict[Path, str] = {}
        self.analysis_complete = False
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, name="VRAMGeist Analysis")
        
        # Settings display
        validation_status = ""
        if self.options.llama_bin:
            validation_status = f" | Validation: {'ON' if self.options.validate_settings else 'OFF'}"
        else:
            validation_status = " | Validation: UNAVAILABLE"
        
        settings_text = f"Mode: {self.options.optimize_for.upper()} | Debug: {'ON' if self.options.debug else 'OFF'}{validation_status} | Keys: 'v'=toggle validation, 'q'=quit"
        yield Static(settings_text, id="settings", classes="settings")
        
        # Results area
        with Vertical(id="results-container"):
            yield Static("Starting analysis...", id="results", classes="results analyzing")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Start analysis when the app mounts."""
        await self.start_analysis()
    
    async def start_analysis(self) -> None:
        """Analyze all files and display results."""
        results_widget = self.query_one("#results")
        
        if not self.paths:
            results_widget.update("No files to analyze.")
            return
        
        # Show initial status
        file_list = "\n".join([f"⏳ Analyzing: {path.name}" for path in self.paths])
        results_widget.update(f"Analyzing {len(self.paths)} files:\n\n{file_list}")
        
        # Analyze each file
        for i, path in enumerate(self.paths):
            try:
                # Update status
                current_file = f"🔄 Analyzing: {path.name} ({i+1}/{len(self.paths)})"
                completed_files = []
                
                for j, p in enumerate(self.paths):
                    if j < i:
                        if p in self.results:
                            completed_files.append(f"✅ {p.name}: Analysis complete")
                        elif p in self.errors:
                            completed_files.append(f"❌ {p.name}: {self.errors[p]}")
                    elif j == i:
                        completed_files.append(current_file)
                    else:
                        completed_files.append(f"⏳ {p.name}: Waiting")
                
                results_widget.update("\n".join(completed_files))
                
                # Perform analysis
                result = await self.analyze_file(path)
                self.results[path] = result
                
            except Exception as e:
                self.errors[path] = str(e)
        
        # Show final results
        await self.show_final_results()
        self.analysis_complete = True
    
    async def analyze_file(self, path: Path) -> Dict[str, Any]:
        """Analyze a single GGUF file."""
        # Run analysis in thread to avoid blocking UI
        loop = asyncio.get_running_loop()
        
        def run_analysis():
            from ..config import DEFAULT_CONFIG
            from .. import ui as ui_module
            
            return ui_module.analyze_gguf_file_with_config(
                str(path),
                config=DEFAULT_CONFIG,
                vram_override=None,
                ram_override=None,
                force_detect=False,
                optimize_for=self.options.optimize_for,
                gpu_bandwidth_gbps=None,
                measure_bandwidth=False,
                measure_tps=False,
                llama_bin=self.options.llama_bin,
                bench_contexts=None,
                debug=self.options.debug,
                validate_settings=self.options.validate_settings,
                validation_timeout=self.options.validation_timeout,
            )
        
        return await loop.run_in_executor(None, run_analysis)
    
    async def show_final_results(self) -> None:
        """Display final analysis results."""
        results_widget = self.query_one("#results")
        
        lines = []
        lines.append(f"[bold green]Analysis Complete![/bold green]")
        lines.append("")
        
        for path in self.paths:
            if path in self.results:
                result = self.results[path]
                lines.append(f"[bold cyan]📄 {path.name}[/bold cyan]")
                
                # Show recommendation
                rec = result.get("recommendation", {})
                if rec:
                    gpu_layers = rec.get("gpu_layers", "?")
                    max_context = rec.get("max_context", "?")
                    expected_vram = rec.get("expected_vram_mb", 0)
                    expected_ram = rec.get("expected_ram_mb", 0)
                    
                    lines.append(f"  Recommended: {gpu_layers} GPU layers, {max_context:,} context")
                    lines.append(f"  Memory: {expected_vram:.0f} MB VRAM, {expected_ram:.0f} MB RAM")
                    
                    # Show validation results
                    validation = result.get("validation")
                    if validation:
                        if validation.get("validated"):
                            lines.append(f"  [green]🔒 Validation: PASSED[/green]")
                        else:
                            reason = validation.get("reason", "Unknown error")
                            lines.append(f"  [red]⚠️  Validation: FAILED ({reason})[/red]")
                            
                            # Show recommendations
                            recommendations = validation.get("recommendations", [])
                            for rec_text in recommendations:
                                lines.append(f"    {rec_text}")
                    elif self.options.validate_settings:
                        lines.append(f"  [yellow]⚠️  Validation: Not performed[/yellow]")
                
                lines.append("")
                
            elif path in self.errors:
                lines.append(f"[red]❌ {path.name}: {self.errors[path]}[/red]")
                lines.append("")
        
        lines.append("[dim]Press 'q' to quit, 'v' to toggle validation[/dim]")
        results_widget.update("\n".join(lines))
    
    async def on_key(self, event) -> None:
        """Handle keyboard input."""
        if event.key == "q":
            await self.action_quit()
        
        elif event.key == "v":
            # Toggle validation
            self.options.validate_settings = not self.options.validate_settings
            
            # Update settings display
            validation_status = ""
            if self.options.llama_bin:
                validation_status = f" | Validation: {'ON' if self.options.validate_settings else 'OFF'}"
            else:
                validation_status = " | Validation: UNAVAILABLE"
            
            settings_text = f"Mode: {self.options.optimize_for.upper()} | Debug: {'ON' if self.options.debug else 'OFF'}{validation_status} | Keys: 'v'=toggle validation, 'q'=quit"
            settings_widget = self.query_one("#settings")
            settings_widget.update(settings_text)
            
            # If analysis is complete, re-run with new validation setting
            if self.analysis_complete:
                self.results.clear()
                self.errors.clear()
                self.analysis_complete = False
                await self.start_analysis()


def run_simple_analysis_tui(paths: List[Path], options: TUIOptions) -> int:
    """Run the simple analysis TUI."""
    app = SimpleAnalysisApp(paths, options)
    try:
        app.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1