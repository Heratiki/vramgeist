import pytest
import tempfile
import struct
import os
from pathlib import Path
from src.vramgeist.gguf import estimate_model_size_mb, read_gguf_metadata


class TestGGUFParsing:
    """Test GGUF file parsing functions"""
    
    def test_estimate_model_size_mb_existing_file(self):
        """Test model size estimation with a real file"""
        # Create a temporary file with known size
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            # Write 1MB of data
            f.write(b'x' * (1024 * 1024))
            temp_path = f.name
        
        try:
            result = estimate_model_size_mb(temp_path)
            assert result == 1.0  # Should be exactly 1MB
        finally:
            os.unlink(temp_path)
    
    def test_estimate_model_size_mb_nonexistent_file(self):
        """Test model size estimation with non-existent file"""
        result = estimate_model_size_mb("/nonexistent/file.gguf")
        assert result == 0.0  # Should return 0 for missing files
    
    def create_mock_gguf_file(self, metadata_dict=None):
        """Create a mock GGUF file with basic structure"""
        if metadata_dict is None:
            metadata_dict = {"llama.block_count": 32}
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.gguf', delete=False)
        
        # Write GGUF magic number and version
        temp_file.write(b'GGUF')  # Magic
        temp_file.write(struct.pack('<I', 3))  # Version 3
        
        # Write tensor count and metadata count
        temp_file.write(struct.pack('<Q', 0))  # Tensor count
        temp_file.write(struct.pack('<Q', len(metadata_dict)))  # Metadata count
        
        # Write metadata
        for key, value in metadata_dict.items():
            # Write key length and key
            key_bytes = key.encode('utf-8')
            temp_file.write(struct.pack('<Q', len(key_bytes)))
            temp_file.write(key_bytes)
            
            # Write value type and value (assuming int32 for simplicity)
            if isinstance(value, int):
                temp_file.write(struct.pack('<I', 4))  # INT32 type
                temp_file.write(struct.pack('<i', value))
            elif isinstance(value, str):
                temp_file.write(struct.pack('<I', 8))  # STRING type
                value_bytes = value.encode('utf-8')
                temp_file.write(struct.pack('<Q', len(value_bytes)))
                temp_file.write(value_bytes)
        
        temp_file.close()
        return temp_file.name
    
    def test_read_gguf_metadata_valid_file(self):
        """Test reading metadata from a valid GGUF file"""
        test_metadata = {
            "llama.block_count": 32,
            "llama.model_name": "test_model"
        }
        
        temp_path = self.create_mock_gguf_file(test_metadata)
        
        try:
            result = read_gguf_metadata(temp_path)
            assert result is not None
            assert "llama.block_count" in result
            assert result["llama.block_count"] == 32
            assert "llama.model_name" in result
            assert result["llama.model_name"] == "test_model"
        finally:
            os.unlink(temp_path)
    
    def test_read_gguf_metadata_nonexistent_file(self):
        """Test reading metadata from non-existent file"""
        result = read_gguf_metadata("/nonexistent/file.gguf")
        assert result is None
    
    def test_read_gguf_metadata_invalid_magic(self):
        """Test reading metadata from file with invalid magic number"""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'INVALID_MAGIC')
            temp_path = f.name
        
        try:
            result = read_gguf_metadata(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)
    
    def test_read_gguf_metadata_empty_file(self):
        """Test reading metadata from empty file"""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            temp_path = f.name
        
        try:
            result = read_gguf_metadata(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)
    
    def test_read_gguf_metadata_truncated_file(self):
        """Test reading metadata from truncated file"""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'GGUF')  # Only magic, missing version and other data
            temp_path = f.name
        
        try:
            result = read_gguf_metadata(temp_path)
            assert result is None  # Should handle truncated files gracefully
        finally:
            os.unlink(temp_path)
    
    def test_path_handling(self):
        """Test that both string and Path objects work"""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'x' * 1024)  # 1KB
            temp_path = f.name
        
        try:
            # Test with string path
            result1 = estimate_model_size_mb(temp_path)
            
            # Test with Path object
            result2 = estimate_model_size_mb(Path(temp_path))
            
            assert result1 == result2
            assert result1 > 0
        finally:
            os.unlink(temp_path)