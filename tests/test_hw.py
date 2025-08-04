from src.vramgeist.hw import _parse_dxdiag_content


class TestHardwareDetection:
    """Test hardware detection functions"""
    
    def test_parse_dxdiag_content_nvidia(self):
        """Test parsing dxdiag content with NVIDIA GPU"""
        sample_content = """
------------------
System Information
------------------
        Time of this report: 1/15/2024, 14:30:25
       Machine name: TEST-PC
   Operating System: Windows 11 Home 64-bit (10.0, Build 22631)
           Language: English (Regional Setting: English)

---------------
Display Devices
---------------
           Card name: NVIDIA GeForce RTX 4090
        Manufacturer: NVIDIA
           Chip type: NVIDIA GeForce RTX 4090
            DAC type: Integrated RAMDAC
         Device type: Full Device (POST)
          Device key: Enum\\PCI\\VEN_10DE&DEV_2684&SUBSYS_408E1043&REV_A1
       Device status: 0180200A [DN][OT][EM]
  Display connector: \\\\?\\DISPLAY#GSM59F2#5&1084e14f&0&UID4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}
      Primary device: Yes
       Video processor: GeForce RTX 4090
         Video memory: 24545 MB
       Dedicated memory: 24545 MB
          Shared memory: 0 MB
           Current mode: 3440 x 1440 (32 bit) (165Hz)
            HDR Support: Supported
        """
        
        result = _parse_dxdiag_content(sample_content)
        assert result == [24545]
    
    def test_parse_dxdiag_content_amd(self):
        """Test parsing dxdiag content with AMD GPU"""
        sample_content = """
Display Devices
---------------
           Card name: AMD Radeon RX 7900 XTX
        Manufacturer: Advanced Micro Devices, Inc.
           Chip type: AMD Radeon Graphics Processor (0x744C)
            DAC type: Internal DAC(400MHz)
         Device type: Full Device
          Device key: Enum\\PCI\\VEN_1002&DEV_744C&SUBSYS_31621787&REV_00
       Device status: 0180200A [DN][OT][EM]
  Display connector: \\\\?\\DISPLAY#SAM0FE7#5&2f6b165f&0&UID4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}
      Primary device: Yes
       Video processor: AMD Radeon RX 7900 XTX
         Video memory: 24564 MB
       Dedicated memory: 24564 MB
          Shared memory: 0 MB
        """
        
        result = _parse_dxdiag_content(sample_content)
        assert result == [24564]
    
    def test_parse_dxdiag_content_intel_integrated(self):
        """Test parsing dxdiag content with Intel integrated graphics"""
        sample_content = """
Display Devices
---------------
           Card name: Intel(R) UHD Graphics 770
        Manufacturer: Intel Corporation
           Chip type: Intel(R) UHD Graphics Family
            DAC type: Internal
         Device type: Full Device
          Device key: Enum\\PCI\\VEN_8086&DEV_A780&SUBSYS_86941043&REV_0C
       Device status: 0180200A [DN][OT][EM]
      Primary device: Yes
       Video processor: Intel(R) UHD Graphics 770
         Video memory: 128 MB
       Dedicated memory: 0 MB
          Shared memory: 15872 MB
        """
        
        result = _parse_dxdiag_content(sample_content)
        # Should filter out small dedicated memory values
        assert result == []  # 128 MB is below our 256 MB threshold
    
    def test_parse_dxdiag_content_multi_gpu(self):
        """Test parsing dxdiag content with multiple GPUs"""
        sample_content = """
Display Devices
---------------
           Card name: NVIDIA GeForce RTX 4090
        Manufacturer: NVIDIA
         Video memory: 24545 MB
       Dedicated memory: 24545 MB

           Card name: NVIDIA GeForce RTX 3080
        Manufacturer: NVIDIA  
         Video memory: 10240 MB
       Dedicated memory: 10240 MB
        """
        
        result = _parse_dxdiag_content(sample_content)
        # Should return both, sorted by size (largest first)
        assert result == [24545, 10240]
    
    def test_parse_dxdiag_content_gb_values(self):
        """Test parsing dxdiag content with GB values"""
        sample_content = """
Display Devices
---------------
           Card name: NVIDIA GeForce RTX 4080
        Manufacturer: NVIDIA
         Video memory: 16.0 GB
       Dedicated memory: 16.0 GB
        """
        
        result = _parse_dxdiag_content(sample_content)
        assert result == [16384]  # 16 GB = 16384 MB
    
    def test_parse_dxdiag_content_no_gpu(self):
        """Test parsing dxdiag content with no valid GPU"""
        sample_content = """
Display Devices
---------------
           Card name: Microsoft Basic Display Adapter
        Manufacturer: Microsoft Corporation
         Video memory: 0 MB
       Dedicated memory: 0 MB
          Shared memory: 0 MB
        """
        
        result = _parse_dxdiag_content(sample_content)
        assert result == []
    
    def test_parse_dxdiag_content_empty(self):
        """Test parsing empty or invalid dxdiag content"""
        result = _parse_dxdiag_content("")
        assert result == []
        
        result = _parse_dxdiag_content("Invalid content with no GPU info")
        assert result == []

    def test_parse_dxdiag_content_single_gpu_dedicated_and_total_prefers_dedicated(self):
        """Single GPU lists both Dedicated and Total; parser should prefer Dedicated only."""
        sample_content = """
Display Devices
---------------
           Card name: NVIDIA GeForce RTX 3070
        Manufacturer: NVIDIA
         Dedicated memory: 8192 MB
         Total memory: 16384 MB
         Display memory: 16384 MB
        """
        result = _parse_dxdiag_content(sample_content)
        assert result == [8192]

    def test_parse_dxdiag_content_integrated_large_shared_ignored(self):
        """Integrated GPU with large shared memory but tiny/zero dedicated should not inflate VRAM."""
        sample_content = """
Display Devices
---------------
           Card name: Intel(R) Iris Xe Graphics
        Manufacturer: Intel Corporation
         Dedicated memory: 0 MB
         Video memory: 128 MB
         Approx. Total Memory: 16384 MB
        """
        result = _parse_dxdiag_content(sample_content)
        # No Tier A or B above threshold, Tier C should be ignored due to precedence
        assert result == []