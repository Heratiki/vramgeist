import os
import struct
from typing import Dict, Any, List, Tuple, Optional
from rich.console import Console

console = Console(force_terminal=True, width=120)


class ParserError(Exception):
    pass


def read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ParserError("Unexpected EOF while reading GGUF")
    return data


def read_gguf_metadata(filepath: str, max_kv: int = 1000, max_key_len: int = 1024, max_str_len: int = 1_000_000) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Extract basic metadata from GGUF file with defensive parsing.
    Returns (metadata_dict or None on severe mismatch, warnings_list).
    """
    warnings: List[str] = []
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                warnings.append("Invalid magic header, not a GGUF file")
                return None, warnings

            _ = struct.unpack("<I", read_exact(f, 4))[0]  # version (unused)
            _ = struct.unpack("<Q", read_exact(f, 8))[0]  # tensor_count (unused)
            metadata_kv_count = struct.unpack("<Q", read_exact(f, 8))[0]

            if metadata_kv_count > max_kv:
                warnings.append(f"metadata_kv_count {metadata_kv_count} exceeds limit {max_kv}, truncating")
                metadata_kv_count = max_kv

            metadata: Dict[str, Any] = {}
            for _ in range(metadata_kv_count):
                key_len = struct.unpack("<Q", read_exact(f, 8))[0]
                if key_len > max_key_len:
                    warnings.append(f"key_len {key_len} exceeds limit {max_key_len}, stopping parse")
                    break
                key = read_exact(f, key_len).decode("utf-8", errors="replace")

                value_type = struct.unpack("<I", read_exact(f, 4))[0]

                if value_type == 4:  # STRING
                    value_len = struct.unpack("<Q", read_exact(f, 8))[0]
                    if value_len > max_str_len:
                        warnings.append(f"string value_len {value_len} exceeds limit {max_str_len}, truncating")
                        value_len = max_str_len
                    value = read_exact(f, value_len).decode("utf-8", errors="replace")
                elif value_type == 5:  # ARRAY
                    array_type = struct.unpack("<I", read_exact(f, 4))[0]
                    array_len = struct.unpack("<Q", read_exact(f, 8))[0]
                    # Skip array payload in a length-aware way
                    if array_type == 4:  # STRING array
                        for _ in range(array_len):
                            str_len = struct.unpack("<Q", read_exact(f, 8))[0]
                            # clamp reads for safety
                            if str_len > max_str_len:
                                warnings.append(f"string element len {str_len} exceeds limit {max_str_len}, truncating")
                                str_len = max_str_len
                            _ = read_exact(f, str_len)
                        value = f"[array of {array_len} strings]"
                    else:
                        # For non-string arrays, we only note the length
                        value = f"[array of {array_len} items]"
                        # NOTE: skipping raw payload requires knowing element width; keeping it abstract
                elif value_type in (6, 7):  # INT32, UINT32
                    value = struct.unpack("<I", read_exact(f, 4))[0]
                elif value_type in (8, 9):  # INT64, UINT64
                    value = struct.unpack("<Q", read_exact(f, 8))[0]
                elif value_type in (10, 11):  # FLOAT32, FLOAT64
                    if value_type == 10:
                        value = struct.unpack("<f", read_exact(f, 4))[0]
                    else:
                        value = struct.unpack("<d", read_exact(f, 8))[0]
                elif value_type == 12:  # BOOL
                    value = struct.unpack("<?", read_exact(f, 1))[0]
                else:
                    # Unknown type: cannot determine payload size; emit warning and abort kv parsing safely
                    warnings.append(f"Unknown value_type {value_type}, stopping metadata parse")
                    break

                metadata[key] = value

            return metadata, warnings
    except ParserError as pe:
        console.print(f"[yellow]GGUF parse warning: {pe}[/yellow]")
        return None, [f"parse_error: {pe}"]
    except Exception as e:
        console.print(f"[red]Error reading GGUF metadata: {e}[/red]")
        return None, [f"error: {e}"]


def estimate_model_size_mb(filepath: str) -> float:
    try:
        file_size = os.path.getsize(filepath)
        return file_size / (1024 * 1024)
    except Exception:
        return 0.0