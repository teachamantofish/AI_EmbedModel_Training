#!/usr/bin/env python3
"""
FORCE GPU ACTIVITY TEST
======================

This test will force GPU operations to verify your RX 580 can be engaged.
It simulates what should happen during training.
"""

import torch
import time
import psutil

def force_gpu_activity():
    """Force sustained GPU activity to verify RX 580 works."""
    print("🔥 FORCING GPU ACTIVITY TEST")
    print("=" * 50)
    print("Watch Task Manager - GPU tab!")
    print("This test will run GPU operations for 30 seconds...")
    
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"✅ DirectML device: {device}")
    except Exception as e:
        print(f"❌ DirectML failed: {e}")
        return False
    
    print("\n🚀 Starting sustained GPU workload...")
    print("If your RX 580 is working, you should see GPU activity in Task Manager!")
    
    start_time = time.time()
    operation_count = 0
    
    try:
        # Run sustained GPU operations for 30 seconds
        while time.time() - start_time < 30:
            # Large matrix operations that should stress the GPU
            size = 1024
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Multiple operations per loop
            for _ in range(10):
                c = torch.matmul(a, b)
                c = torch.relu(c)
                c = torch.matmul(c, a)
                operation_count += 3
            
            # Clean up tensors to prevent memory issues
            del a, b, c
            
            # Progress indicator
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                print(f"   {int(elapsed)}s: {operation_count} operations completed")
                time.sleep(0.1)  # Prevent multiple prints per second
    
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False
    
    total_time = time.time() - start_time
    ops_per_second = operation_count / total_time
    
    print(f"\n✅ SUSTAINED GPU TEST COMPLETED")
    print(f"   Duration: {total_time:.1f} seconds")
    print(f"   Operations: {operation_count}")
    print(f"   Performance: {ops_per_second:.1f} ops/second")
    
    if ops_per_second > 100:
        print("🎉 Your RX 580 is working well!")
        print("GPU should have shown activity in Task Manager")
    else:
        print("⚠️  Performance seems low - check if GPU was actually used")
    
    return True

def test_model_gpu_placement():
    """Test if we can get a simple model to use GPU."""
    print("\n🧠 TESTING MODEL GPU PLACEMENT")
    print("=" * 50)
    
    try:
        import torch_directml
        device = torch_directml.device()
        
        # Create a simple neural network and move to GPU
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ).to(device)
        
        print(f"✅ Model moved to device: {device}")
        
        # Test inference with the model
        batch_size = 32
        input_tensor = torch.randn(batch_size, 768, device=device)
        
        print("🔄 Running model inference on GPU...")
        start_time = time.time()
        
        # Run multiple forward passes
        for i in range(100):
            with torch.no_grad():
                output = model(input_tensor)
        
        inference_time = time.time() - start_time
        
        print(f"✅ Model inference completed: {output.shape}")
        print(f"   100 forward passes in {inference_time:.2f}s")
        print(f"   Average: {inference_time*10:.1f}ms per pass")
        
        return True
        
    except Exception as e:
        print(f"❌ Model GPU test failed: {e}")
        return False

if __name__ == "__main__":
    print("COMPREHENSIVE GPU ACTIVITY TEST")
    print("This will force your RX 580 to work hard!")
    print("Keep Task Manager open on the GPU tab...")
    
    # Test 1: Force GPU activity
    success1 = force_gpu_activity()
    
    # Test 2: Model placement
    success2 = test_model_gpu_placement()
    
    if success1 and success2:
        print("\n🎉 GPU TESTS SUCCESSFUL!")
        print("Your RX 580 is working and should have shown activity")
        print("The training should now use GPU acceleration")
    else:
        print("\n❌ GPU tests failed")
        print("Training will fall back to CPU only")