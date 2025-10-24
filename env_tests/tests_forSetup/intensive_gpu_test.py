#!/usr/bin/env python3
"""
INTENSIVE GPU TEST - Force RX 580 Activity
==========================================

This test directly forces intensive operations on your RX 580
to definitively show GPU activity in Task Manager.
"""

import torch
import torch_directml
import time


def intensive_gpu_test():
    """Run intensive GPU operations that will definitely show in Task Manager."""
    print("🔥 INTENSIVE GPU TEST - Forcing RX 580 Activity")
    print("=" * 60)
    print("This will stress test your GPU for 30 seconds")
    print("🎯 WATCH TASK MANAGER > PERFORMANCE > GPU NOW!")
    print()
    
    # Get DirectML device
    device = torch_directml.device()
    print(f"Device: {device}")
    
    # Create large tensors that will use significant GPU resources
    print("Creating large tensors on GPU...")
    size = 3000  # Larger tensors for more GPU usage
    
    # Multiple large tensors
    tensors = []
    for i in range(5):  # Create 5 large tensors
        tensor = torch.randn(size, size, device=device, dtype=torch.float32)
        tensors.append(tensor)
        print(f"  Tensor {i+1}: {tensor.shape} ({tensor.nbytes/1024**2:.1f} MB)")
    
    total_memory = sum(t.nbytes for t in tensors) / 1024**2
    print(f"📊 Total GPU memory allocated: {total_memory:.1f} MB")
    
    input("\n⏸️  Press ENTER when you're watching Task Manager GPU...")
    
    print("\n🚀 STARTING 30-SECOND GPU STRESS TEST")
    print("GPU should show HIGH activity now!")
    
    start_time = time.time()
    operation_count = 0
    
    try:
        while time.time() - start_time < 30:
            # Intensive matrix operations
            for i in range(len(tensors)-1):
                # Matrix multiplication (very GPU intensive)
                result = torch.matmul(tensors[i], tensors[i+1])
                
                # More operations to keep GPU busy
                result = torch.relu(result)
                result = torch.sigmoid(result)
                result = torch.log(result + 1e-8)  # Add small value to avoid log(0)
                
                # Replace one tensor to keep memory active
                tensors[i] = result
                
                operation_count += 1
            
            # Print progress
            elapsed = time.time() - start_time
            if operation_count % 20 == 0:
                print(f"   ⚡ {elapsed:.1f}s - {operation_count} operations - CHECK GPU USAGE!")
    
    except Exception as e:
        print(f"Error during stress test: {e}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Stress test completed in {total_time:.1f} seconds")
    print(f"📈 Total operations: {operation_count}")
    print(f"🏃 Operations per second: {operation_count/total_time:.1f}")
    
    print("\n🎯 RESULTS:")
    print("   ✅ If GPU usage showed up in Task Manager = RX 580 working!")
    print("   ❌ If no GPU activity = DirectML/driver issue")
    
    # Cleanup
    del tensors
    print("\n🧹 GPU memory cleaned up")

if __name__ == "__main__":
    intensive_gpu_test()