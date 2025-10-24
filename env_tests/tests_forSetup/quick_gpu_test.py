#!/usr/bin/env python3
"""
QUICK GPU TEST (< 30 seconds)
=============================

Fast lightweight test to verify your Radeon RX 580 is working with DirectML.
No model loading, no training - just pure GPU validation.
"""

import time
import torch
import psutil
import os

def quick_gpu_test():
    """Ultra-fast GPU test - should complete in under 30 seconds."""
    print("🚀 QUICK GPU TEST - Radeon RX 580 Verification")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Check DirectML availability (should be instant)
    print("1. Checking DirectML availability...")
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"   ✅ DirectML device detected: {device}")
    except Exception as e:
        print(f"   ❌ DirectML failed: {e}")
        return False
    
    # Step 2: Test small GPU operations (< 5 seconds)
    print("2. Testing basic GPU operations...")
    try:
        # Small tensor operations
        for size in [100, 500, 1000]:
            test_tensor = torch.randn(size, size, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"   ✅ {size}x{size} matrix multiplication successful")
        
        # Test GPU memory allocation
        memory_test = torch.randn(2000, 2000, device=device)
        print(f"   ✅ GPU memory allocation successful: {memory_test.nbytes / 1024**2:.1f} MB")
        del memory_test  # Free memory
        
    except Exception as e:
        print(f"   ❌ GPU operations failed: {e}")
        return False
    
    # Step 3: Test embeddings-like operations (< 10 seconds)  
    print("3. Testing embedding-style operations...")
    try:
        # Simulate embedding operations without loading actual models
        batch_size = 32
        embedding_dim = 768
        seq_length = 512
        
        # Simulate transformer-like operations
        inputs = torch.randn(batch_size, seq_length, embedding_dim, device=device)
        weights = torch.randn(embedding_dim, embedding_dim, device=device)
        
        # Matrix multiplication (core of transformer operations)
        output = torch.matmul(inputs, weights)
        
        # Activation function
        output = torch.relu(output)
        
        # Pooling (typical in sentence transformers)
        pooled = torch.mean(output, dim=1)  # Should be [32, 768]
        
        print(f"   ✅ Embedding simulation successful: {pooled.shape}")
        print(f"   ✅ Processed batch of {batch_size} sequences")
        
    except Exception as e:
        print(f"   ❌ Embedding operations failed: {e}")
        return False
    
    # Step 4: Performance benchmark (< 5 seconds)
    print("4. Quick performance benchmark...")
    try:
        benchmark_start = time.time()
        
        # Run 100 small operations to test sustained performance
        for i in range(100):
            a = torch.randn(256, 256, device=device)
            b = torch.randn(256, 256, device=device)
            c = torch.matmul(a, b)
        
        benchmark_time = time.time() - benchmark_start
        ops_per_second = 100 / benchmark_time
        
        print(f"   ✅ Benchmark: {ops_per_second:.1f} operations/second")
        print(f"   ✅ Average time per operation: {benchmark_time*10:.1f} ms")
        
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
        return False
    
    # Step 5: System resource check
    print("5. System resource check...")
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        print(f"   ✅ CPU usage: {cpu_percent}%")
        print(f"   ✅ RAM usage: {memory_info.percent}% ({memory_info.used/1024**3:.1f}GB/{memory_info.total/1024**3:.1f}GB)")
        
    except Exception as e:
        print(f"   ❌ System check failed: {e}")
    
    total_time = time.time() - start_time
    
    print("=" * 50)
    print(f"🎉 QUICK TEST COMPLETED in {total_time:.1f} seconds")
    print(f"✅ Your Radeon RX 580 is working with DirectML!")
    print(f"✅ Ready for embedding model training")
    
    # Estimate training performance
    estimated_training_speed = ops_per_second * 0.1  # Very rough estimate
    print(f"📊 Estimated training performance: ~{estimated_training_speed:.0f} samples/minute")
    
    return True

if __name__ == "__main__":
    success = quick_gpu_test()
    if success:
        print("\n💡 Next steps:")
        print("   - Run: python embedding_finetuner.py --action train")
        print("   - Your RX 580 will accelerate the training process")
    else:
        print("\n❌ GPU test failed - check DirectML installation")