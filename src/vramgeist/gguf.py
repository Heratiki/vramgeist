import os
import re
import struct
from typing import Dict, Any, List, Tuple, Optional
from ._rich_fallback import Console

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
    Extract basic metadata from a GGUF file with defensive parsing.
    Returns (metadata_dict or None on severe mismatch, warnings_list).
    Enhancements: string arrays are returned as Python lists and we attempt to
    infer key=value pairs from string blobs/arrays (useful for HF-style metadata).
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

                # Value type mapping comes from GGUF spec; handle common types defensively
                if value_type == 4:  # STRING
                    value_len = struct.unpack("<Q", read_exact(f, 8))[0]
                    if value_len > max_str_len:
                        warnings.append(f"string value_len {value_len} exceeds limit {max_str_len}, truncating")
                        value_len = max_str_len
                    value = read_exact(f, value_len).decode("utf-8", errors="replace")
                elif value_type == 5:  # ARRAY
                    array_type = struct.unpack("<I", read_exact(f, 4))[0]
                    array_len = struct.unpack("<Q", read_exact(f, 8))[0]
                    if array_type == 4:  # STRING array
                        items: List[str] = []
                        for _ in range(array_len):
                            str_len = struct.unpack("<Q", read_exact(f, 8))[0]
                            if str_len > max_str_len:
                                warnings.append(f"string element len {str_len} exceeds limit {max_str_len}, truncating")
                                str_len = max_str_len
                            raw = read_exact(f, str_len)
                            try:
                                s = raw.decode("utf-8", errors="replace")
                            except Exception:
                                s = raw.decode("latin-1", errors="replace")
                            items.append(s)
                        value = items
                    else:
                        # For non-string arrays we don't try to decode elements; note length
                        value = f"[array of {array_len} items]"
                        # Attempt to skip payload conservatively: if element size known, could skip
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
                    warnings.append(f"Unknown value_type {value_type}, stopping metadata parse")
                    break

                metadata[key] = value

            # Post-process collected metadata: infer key=value pairs from strings/lists
            def _try_parse_literal(s: str):
                s_lower = s.lower()
                if s_lower in ("true", "false"):
                    return s_lower == "true"
                try:
                    if "." in s:
                        return float(s)
                    return int(s)
                except Exception:
                    return s

            for k, v in list(metadata.items()):
                try:
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and "=" in item:
                                subk, subv = item.split("=", 1)
                                subk = subk.strip()
                                subv = subv.strip()
                                if subk and subk not in metadata:
                                    metadata[subk] = _try_parse_literal(subv)
                    elif isinstance(v, str) and "=" in v:
                        parts = [p.strip() for p in re.split(r'[\n,;]', v) if p.strip()]
                        for part in parts:
                            if "=" in part:
                                subk, subv = part.split("=", 1)
                                subk = subk.strip()
                                if subk and subk not in metadata:
                                    metadata[subk] = _try_parse_literal(subv.strip())
                except Exception:
                    continue

            return metadata, warnings
    except ParserError as pe:
        console.print(f"[yellow]GGUF parse warning: {pe}[/yellow]")
        return None, [f"parse_error: {pe}"]
    except Exception as e:
        console.print(f"[red]Error reading GGUF metadata: {e}[/red]")
        return None, [f"error: {e}"]


def estimate_model_size_mb(filepath: str | os.PathLike) -> float:
    """Estimate model file size in MB. Accepts string paths or Path objects."""
    try:
        # Support pathlib.Path inputs
        if hasattr(filepath, 'as_posix') or hasattr(filepath, '__fspath__'):
            filepath = os.fspath(filepath)
        file_size = os.path.getsize(filepath)
        return file_size / (1024 * 1024)
    except Exception:
        return 0.0