"""
Configuration persistence for VRAMGeist settings.

Stores user preferences like llama.cpp binary path to improve user experience
across sessions.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_config_dir() -> Path:
    """Get the user configuration directory for VRAMGeist"""
    # Use platform-appropriate config directory
    if os.name == 'nt':  # Windows
        config_base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:  # Unix-like (Linux, macOS)
        config_base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    
    config_dir = config_base / 'vramgeist'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the configuration file path"""
    return get_config_dir() / 'config.json'


def load_config() -> Dict[str, Any]:
    """Load configuration from file, returning empty dict if not found or invalid"""
    config_file = get_config_file()
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # If config is corrupted, return empty config
        return {}


def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file, returning True on success"""
    try:
        config_file = get_config_file()
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        return True
    except (OSError, UnicodeEncodeError):
        return False


def get_llama_bin_path() -> Optional[str]:
    """Get the stored llama.cpp binary path if it exists and is valid"""
    config = load_config()
    llama_bin = config.get('llama_bin_path')
    
    if llama_bin and Path(llama_bin).exists():
        return llama_bin
    return None


def save_llama_bin_path(llama_bin: str) -> bool:
    """Save a validated llama.cpp binary path to config"""
    if not llama_bin or not Path(llama_bin).exists():
        return False
    
    config = load_config()
    config['llama_bin_path'] = str(Path(llama_bin).resolve())
    config['llama_bin_last_verified'] = str(Path(llama_bin).stat().st_mtime)
    return save_config(config)


def validate_and_update_llama_bin() -> Optional[str]:
    """Validate stored llama.cpp path and return it if still valid, None otherwise"""
    config = load_config()
    llama_bin = config.get('llama_bin_path')
    
    if not llama_bin:
        return None
        
    llama_path = Path(llama_bin)
    if not llama_path.exists():
        # Remove invalid path from config
        config.pop('llama_bin_path', None)
        config.pop('llama_bin_last_verified', None)
        save_config(config)
        return None
    
    return llama_bin


def get_validation_timeout() -> float:
    """Get stored validation timeout, default to 30.0"""
    config = load_config()
    return config.get('validation_timeout', 30.0)


def save_validation_timeout(timeout: float) -> bool:
    """Save validation timeout to config"""
    config = load_config()
    config['validation_timeout'] = timeout
    return save_config(config)


def should_enable_validation_by_default() -> bool:
    """Check if validation should be enabled by default (has working llama.cpp path)"""
    return get_llama_bin_path() is not None


def clear_llama_bin_path() -> bool:
    """Clear the saved llama.cpp binary path from config"""
    config = load_config()
    if 'llama_bin_path' in config:
        config.pop('llama_bin_path', None)
        config.pop('llama_bin_last_verified', None)
        return save_config(config)
    return True


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration for display purposes"""
    config = load_config()
    llama_bin = get_llama_bin_path()
    
    return {
        'config_file': str(get_config_file()),
        'has_llama_bin': llama_bin is not None,
        'llama_bin_path': llama_bin,
        'validation_timeout': get_validation_timeout(),
        'config_exists': get_config_file().exists()
    }