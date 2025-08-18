import tempfile
import struct
import os
from src.vramgeist.gguf import read_gguf_metadata


def create_gguf_with_types(tmp_path):
    p = tmp_path / 'test.gguf'
    with open(p, 'wb') as f:
        f.write(b'GGUF')
        f.write(struct.pack('<I', 3))  # version
        f.write(struct.pack('<Q', 0))  # tensor count
        # We'll write 4 kv pairs
        f.write(struct.pack('<Q', 4))

        # INT32 key
        key = b'int_key'
        f.write(struct.pack('<Q', len(key)))
        f.write(key)
        f.write(struct.pack('<I', 6))  # INT32 type code (6/7 in file)
        f.write(struct.pack('<I', 1234))

        # STRING key
        key = b'str_key'
        f.write(struct.pack('<Q', len(key)))
        f.write(key)
        f.write(struct.pack('<I', 4))  # STRING type
        val = b'hello=1.23;flag=true'
        f.write(struct.pack('<Q', len(val)))
        f.write(val)

        # STRING ARRAY key
        key = b'arr_key'
        f.write(struct.pack('<Q', len(key)))
        f.write(key)
        f.write(struct.pack('<I', 5))  # ARRAY
        f.write(struct.pack('<I', 4))  # array of strings
        f.write(struct.pack('<Q', 2))  # 2 elements
        s1 = b'hidden_size=4096'
        s2 = b'n_layers=32'
        f.write(struct.pack('<Q', len(s1)))
        f.write(s1)
        f.write(struct.pack('<Q', len(s2)))
        f.write(s2)

        # FLOAT64 key
        key = b'flt_key'
        f.write(struct.pack('<Q', len(key)))
        f.write(key)
        f.write(struct.pack('<I', 11))  # FLOAT64
        f.write(struct.pack('<d', 3.14159))
    return str(p)


def test_read_gguf_metadata_various_types(tmp_path):
    p = create_gguf_with_types(tmp_path)
    metadata, warnings = read_gguf_metadata(p)
    assert isinstance(metadata, dict)
    # Parsed ints/floats/derived keys
    assert 'int_key' in metadata
    assert metadata.get('int_key') == 1234
    assert 'flt_key' in metadata
    flt_val = metadata.get('flt_key')
    assert flt_val is not None
    assert abs(float(flt_val) - 3.14159) < 1e-6
    assert 'hidden_size' in metadata
    hidden_val = metadata.get('hidden_size')
    assert hidden_val is not None
    assert int(hidden_val) == 4096
    assert 'n_layers' in metadata
    n_layers_val = metadata.get('n_layers')
    assert n_layers_val is not None
    assert int(n_layers_val) == 32
    # string parsing may produce warnings or metadata keys
    assert isinstance(warnings, list)