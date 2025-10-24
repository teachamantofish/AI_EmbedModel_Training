#!/usr/bin/env python3
"""
GPU Detection and Verification Script
=====================================

This script checks what GPUs are available to PyTorch and tests if your
external Radeon RX 580 is being detected and can be used for computation.
"""

import torch
import platform
import sys

def check_system_info():
    """Display basic system information."""
    print("=== SYSTEM INFORMATION ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print()

def check_cuda():
    """Check CUDA availability (for NVIDIA GPUs)."""
    print("=== CUDA INFORMATION ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices detected")
    print()

def check_directml():
    """Check DirectML availability (for AMD GPUs on Windows)."""
    print("=== DIRECTML INFORMATION ===")
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"✅ DirectML available: {device}")
        
        # Test basic operations
        print("Testing DirectML with basic tensor operations...")
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"✅ DirectML tensor operations successful: {result.shape}")
        
        # Check if this is your RX 580
        print(f"DirectML device: {device}")
        return True
        
    except ImportError:
        print("❌ torch_directml not installed")
        print("Install with: pip install torch-directml")
        return False
    except Exception as e:
        print(f"❌ DirectML error: {e}")
        return False

def check_mps():
    """Check MPS availability (for Apple Silicon)."""
    print("=== MPS INFORMATION ===")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) available")
        return True
    else:
        print("❌ MPS not available (not on Apple Silicon)")
        return False

def check_cpu():
    """Check CPU information."""
    print("=== CPU INFORMATION ===")
    print(f"CPU device available: True")
    
    # Test CPU tensor operations
    test_tensor = torch.randn(100, 100)
    result = torch.matmul(test_tensor, test_tensor)
    print(f"✅ CPU tensor operations successful: {result.shape}")
    print()

def main():
    """Run all GPU detection checks."""
    print("GPU DETECTION AND VERIFICATION")
    print("=" * 50)
    print()
    
    check_system_info()
    check_cuda()
    directml_available = check_directml()
    mps_available = check_mps()
    check_cpu()
    
    print("=== RECOMMENDATION ===")
    if directml_available:
        print("✅ DirectML detected - Your AMD Radeon RX 580 should be usable!")
        print("Use this in your embedding training:")
        print("  - Set device = torch_directml.device()")
        print("  - Move tensors to this device for GPU acceleration")
    elif torch.cuda.is_available():
        print("✅ CUDA detected - NVIDIA GPU available")
    elif mps_available:
        print("✅ MPS detected - Apple Silicon GPU available")
    else:
        print("❌ No GPU acceleration detected")
        print("Recommendations:")
        print("  1. Install torch-directml for AMD GPU: pip install torch-directml")
        print("  2. Update GPU drivers")
        print("  3. Check if RX 580 is properly connected and powered")

if __name__ == "__main__":
    main()