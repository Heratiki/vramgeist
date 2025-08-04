# Minimal TUI package init to avoid import side-effects when extras are missing.
# Do not import textual/rich here; keep lazy imports inside modules that need them.

__all__ = [
    "options",
    "state",
    "app",
]