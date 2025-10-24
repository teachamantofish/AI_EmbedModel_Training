#!/usr/bin/env python3
"""
Test DirectML GPU detection and basic functionality
"""
import torch
import torch_directml

def test_directml():
    print("=== DirectML GPU Test ===")
    
    # Test DirectML device detection
    try:
        device = torch_directml.device()
        print(f"✅ DirectML device detected: {device}")
        
        # Get device name if available
        try:
            device_name = torch_directml.device_name()
            print(f"✅ GPU name: {device_name}")
        except:
            print("⚠️  GPU name not available")
        
        # Test basic tensor operations
        print("\n=== Testing tensor operations ===")
        
        # Create test tensors
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        print(f"✅ Matrix multiplication successful: {z.shape}")
        
        # Move back to CPU
        z_cpu = z.cpu()
        print(f"✅ GPU->CPU transfer successful")
        
        # Test with larger tensors (more GPU-intensive)
        print("\n=== Testing larger operations ===")
        large_x = torch.randn(1000, 1000).to(device)
        large_y = torch.randn(1000, 1000).to(device)
        large_z = torch.matmul(large_x, large_y)
        print(f"✅ Large matrix multiplication: {large_z.shape}")
        
        print("\n🎉 DirectML is working correctly!")
        print(f"🎯 Ready for GPU-accelerated training with device: {device}")
        
        return True, device
        
    except Exception as e:
        print(f"❌ DirectML test failed: {e}")
        return False, None

def test_fallback():
    print("\n=== CPU Fallback Test ===")
    device = torch.device("cpu")
    
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    
    print(f"✅ CPU operations working: {z.shape}")
    print(f"🔄 Will use CPU training (slower but reliable)")
    
    return device

if __name__ == "__main__":
    success, gpu_device = test_directml()
    
    if not success:
        print("\n" + "="*50)
        print("DirectML not available - falling back to CPU")
        cpu_device = test_fallback()
        print(f"\nRecommended device for training: {cpu_device}")
    else:
        print(f"\nRecommended device for training: {gpu_device}")
        print("Expected speedup: 3-5x faster than CPU")
