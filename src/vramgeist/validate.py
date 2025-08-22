"""
Validation module for testing recommended VRAM settings with actual llama.cpp runs.

This module provides functionality to validate that recommended GPU layer and context 
settings actually work in practice by running brief test inferences.
"""
from __future__ import annotations
import subprocess
import tempfile
import shlex
import time
from typing import Optional, Tuple, Dict, Any
import os


def create_validation_prompt(context_length: int) -> str:
    """Create a simple prompt to test the given context length"""
    # Keep the prompt short and simple - we don't need to fill the entire context
    # Just enough to test that the model loads and can generate with the given context size
    # llama.cpp will allocate the full context size regardless of prompt length
    return "Hello! Please generate a short response to test this model configuration."


def validate_llama_cpp_settings(
    model_path: str,
    gpu_layers: int,
    context_length: int,
    llama_bin: str,
    timeout: float = 30.0,
    n_predict: int = 5,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate GPU layers and context settings by running a test inference.
    
    Args:
        model_path: Path to GGUF model file
        gpu_layers: Number of GPU layers to test
        context_length: Context length to test
        llama_bin: Path to llama.cpp binary
        timeout: Timeout in seconds
        n_predict: Number of tokens to generate for test
        
    Returns:
        Tuple of (success, reason, details)
        - success: Whether the test passed
        - reason: Human-readable explanation
        - details: Dict with timing, memory usage, etc.
    """
    if not os.path.exists(llama_bin):
        return False, f"llama.cpp binary not found: {llama_bin}", {}
        
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}", {}
    
    # Ensure minimum context length for validation
    if context_length <= 0:
        context_length = 512  # Use minimum reasonable context
    
    # Create test prompt and write to temporary file to avoid command line length issues
    prompt = create_validation_prompt(context_length)
    
    # Create temporary file for the prompt
    prompt_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(prompt)
            prompt_file = f.name
        
        # Build command using file input instead of direct prompt
        # This significantly reduces command line length
        cmd_template = (
            f'{shlex.quote(llama_bin)} '
            f'-m {shlex.quote(model_path)} '
            f'-f {shlex.quote(prompt_file)} '
            f'-n {n_predict} '
            f'-c {context_length} '
            f'-ngl {gpu_layers} '
            f'--no-display-prompt'
        )
        
    except Exception as e:
        return False, f"Validation failed: Could not create temporary prompt file: {str(e)}", {}
    
    details = {
        "command": cmd_template,
        "gpu_layers": gpu_layers,
        "context_length": context_length,
        "timeout": timeout,
        "n_predict": n_predict
    }
    
    try:
        start_time = time.time()
        
        # Use shlex to properly parse the command and avoid shell issues
        cmd_args = shlex.split(cmd_template)
        
        # Debug: Log the command being executed
        details["debug_command"] = cmd_template
        details["debug_args"] = cmd_args
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        details["elapsed_seconds"] = elapsed
        details["return_code"] = result.returncode
        details["stdout_length"] = len(result.stdout) if result.stdout else 0
        details["stderr_length"] = len(result.stderr) if result.stderr else 0
        details["stdout_sample"] = result.stdout[:200] if result.stdout else ""
        details["stderr_sample"] = result.stderr[:200] if result.stderr else ""
        
        # Check for success indicators
        if result.returncode == 0:
            # Look for successful generation indicators in stdout
            if result.stdout and len(result.stdout.strip()) > 0:
                # Generated some output - this indicates successful loading and generation
                details["generated_text_length"] = len(result.stdout.strip())
                
                # Save successful llama.cpp path for future use
                try:
                    from .config_persist import save_llama_bin_path
                    save_llama_bin_path(llama_bin)
                except ImportError:
                    pass  # Config persistence not available
                
                return True, "Validation successful: model loaded and generated text", details
            else:
                # Check stderr for successful loading indicators even if no stdout
                if result.stderr and ("model params" in result.stderr.lower() or "llama" in result.stderr.lower()):
                    # Also save path for successful loading even without generation
                    try:
                        from .config_persist import save_llama_bin_path
                        save_llama_bin_path(llama_bin)
                    except ImportError:
                        pass
                    return True, "Validation successful: model loaded (no text generation detected)", details
                return False, "Model loaded but failed to generate expected output", details
        
        # Parse common error patterns from stderr
        stderr_lower = result.stderr.lower() if result.stderr else ""
        if "out of memory" in stderr_lower or "cuda out of memory" in stderr_lower:
            return False, "Validation failed: GPU out of memory", details
        elif "failed to allocate" in stderr_lower:
            return False, "Validation failed: Memory allocation failed", details
        elif "context size" in stderr_lower and "too large" in stderr_lower:
            return False, "Validation failed: Context size too large", details
        else:
            return False, f"Validation failed: llama.cpp returned error code {result.returncode}", details
            
    except subprocess.TimeoutExpired:
        details["timed_out"] = True
        return False, f"Validation failed: Operation timed out after {timeout}s", details
    except Exception as e:
        details["exception"] = str(e)
        return False, f"Validation failed: {str(e)}", details
    finally:
        # Clean up temporary prompt file
        if prompt_file and os.path.exists(prompt_file):
            try:
                os.unlink(prompt_file)
            except OSError:
                pass  # Ignore cleanup errors


def create_validation_function(model_path: str, gpu_layers: int, llama_bin: str) -> callable:
    """
    Create a validation function that can be used with the existing validate_context framework.
    
    Returns a function that takes a context length and returns True/False for success.
    """
    def measure_func(context_length: int) -> bool:
        success, reason, details = validate_llama_cpp_settings(
            model_path=model_path,
            gpu_layers=gpu_layers,
            context_length=context_length,
            llama_bin=llama_bin,
            timeout=20.0,  # Shorter timeout for validation
            n_predict=3,   # Fewer tokens for faster validation
        )
        return success
    
    return measure_func


def validate_recommendation(
    model_path: str,
    recommended_gpu_layers: int,
    recommended_context: int,
    llama_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a complete recommendation (GPU layers + context length).
    
    Returns a validation report with success status and recommendations.
    """
    if not llama_bin:
        return {
            "validated": False,
            "reason": "No llama.cpp binary provided for validation",
            "recommendations": ["Provide --llama-bin path to enable validation"]
        }
    
    # Test the exact recommended settings first
    success, reason, details = validate_llama_cpp_settings(
        model_path=model_path,
        gpu_layers=recommended_gpu_layers,
        context_length=recommended_context,
        llama_bin=llama_bin
    )
    
    validation_report = {
        "validated": success,
        "tested_gpu_layers": recommended_gpu_layers,
        "tested_context": recommended_context,
        "reason": reason,
        "details": details,
        "recommendations": []
    }
    
    if success:
        validation_report["recommendations"].append("✅ Recommended settings validated successfully")
        return validation_report
    
    # If validation failed, try to find working settings
    validation_report["recommendations"].append("❌ Initial validation failed")
    
    # Try reducing context length first (more common issue)
    if "memory" in reason.lower() or "context" in reason.lower():
        fallback_contexts = [
            recommended_context // 2,
            recommended_context // 4,
            min(8192, recommended_context // 2),
            4096,
            2048
        ]
        
        for fallback_context in fallback_contexts:
            if fallback_context >= 512:  # Minimum reasonable context
                fallback_success, fallback_reason, fallback_details = validate_llama_cpp_settings(
                    model_path=model_path,
                    gpu_layers=recommended_gpu_layers,
                    context_length=fallback_context,
                    llama_bin=llama_bin
                )
                
                if fallback_success:
                    validation_report["fallback_context"] = fallback_context
                    validation_report["fallback_validated"] = True
                    validation_report["recommendations"].append(
                        f"✅ Fallback validated: try context length {fallback_context:,} instead"
                    )
                    return validation_report
    
    # Try reducing GPU layers if context reduction didn't work
    fallback_gpu_layers = [
        max(0, recommended_gpu_layers - 8),
        max(0, recommended_gpu_layers - 16),
        max(0, recommended_gpu_layers // 2),
        0  # CPU-only fallback
    ]
    
    for fallback_layers in fallback_gpu_layers:
        fallback_success, fallback_reason, fallback_details = validate_llama_cpp_settings(
            model_path=model_path,
            gpu_layers=fallback_layers,
            context_length=min(4096, recommended_context),  # Use smaller context too
            llama_bin=llama_bin
        )
        
        if fallback_success:
            validation_report["fallback_gpu_layers"] = fallback_layers
            validation_report["fallback_context"] = min(4096, recommended_context)
            validation_report["fallback_validated"] = True
            validation_report["recommendations"].append(
                f"✅ Fallback validated: try {fallback_layers} GPU layers with {min(4096, recommended_context):,} context"
            )
            return validation_report
    
    validation_report["recommendations"].append("❌ No working configuration found - consider using a smaller model")
    return validation_report