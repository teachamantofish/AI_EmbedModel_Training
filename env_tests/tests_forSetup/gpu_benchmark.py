#!/usr/bin/env python3
"""
GPU vs CPU BENCHMARK TEST
========================

This test will run the same operations on CPU vs GPU to see if there's
actually a performance difference. If DirectML is working, GPU should
be significantly faster.
"""

import torch
import torch_directml
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_cpu_vs_gpu():
    """Benchmark identical operations on CPU vs GPU."""
    logger.info("🔍 GPU vs CPU BENCHMARK TEST")
    
    # Test parameters - LONGER TEST
    matrix_size = 2000
    num_operations = 300  # 3x longer test
    
    # Setup devices
    cpu_device = torch.device('cpu')
    gpu_device = torch_directml.device()
    
    logger.info(f"✅ CPU device: {cpu_device}")
    logger.info(f"✅ GPU device: {gpu_device}")
    
    # Test 1: CPU Performance
    logger.info(f"\n🐌 CPU TEST - {num_operations} matrix operations ({matrix_size}x{matrix_size})")
    
    cpu_start_time = time.time()
    
    for i in range(num_operations):
        # Create tensors on CPU
        a_cpu = torch.randn(matrix_size, matrix_size, device=cpu_device)
        b_cpu = torch.randn(matrix_size, matrix_size, device=cpu_device)
        
        # Matrix operations on CPU
        c_cpu = torch.matmul(a_cpu, b_cpu)
        d_cpu = torch.relu(c_cpu)
        e_cpu = torch.sigmoid(d_cpu)
        
        # Cleanup
        del a_cpu, b_cpu, c_cpu, d_cpu, e_cpu
        
        if (i + 1) % 50 == 0:  # Less frequent updates for longer test
            logger.info(f"   CPU Progress: {i + 1}/{num_operations}")
    
    cpu_time = time.time() - cpu_start_time
    logger.info(f"📊 CPU Total Time: {cpu_time:.2f} seconds")
    
    # Test 2: GPU Performance
    logger.info(f"\n🚀 GPU TEST - {num_operations} matrix operations ({matrix_size}x{matrix_size})")
    logger.info("📊 WATCH TASK MANAGER GPU NOW - should show activity if working!")
    
    gpu_start_time = time.time()
    
    for i in range(num_operations):
        # Create tensors on GPU
        a_gpu = torch.randn(matrix_size, matrix_size, device=gpu_device)
        b_gpu = torch.randn(matrix_size, matrix_size, device=gpu_device)
        
        # Matrix operations on GPU
        c_gpu = torch.matmul(a_gpu, b_gpu)
        d_gpu = torch.relu(c_gpu)
        e_gpu = torch.sigmoid(d_gpu)
        
        # Cleanup
        del a_gpu, b_gpu, c_gpu, d_gpu, e_gpu
        
        if (i + 1) % 50 == 0:  # Less frequent updates for longer test  
            logger.info(f"   GPU Progress: {i + 1}/{num_operations}")
            logger.info(f"   🎯 Check Task Manager GPU NOW - should show sustained 50% usage!")
    
    gpu_time = time.time() - gpu_start_time
    logger.info(f"📊 GPU Total Time: {gpu_time:.2f} seconds")
    
    # Analysis
    logger.info(f"\n📈 PERFORMANCE COMPARISON:")
    logger.info(f"   CPU Time: {cpu_time:.2f} seconds")
    logger.info(f"   GPU Time: {gpu_time:.2f} seconds")
    
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        logger.info(f"   🚀 GPU Speedup: {speedup:.2f}x faster")
        logger.info(f"   ✅ DirectML is working and accelerating operations!")
        return True
    else:
        slowdown = gpu_time / cpu_time
        logger.info(f"   🐌 GPU Slowdown: {slowdown:.2f}x slower")
        logger.info(f"   ❌ DirectML is NOT working - operations falling back to CPU")
        return False

def test_gpu_memory_allocation():
    """Test if we can actually allocate memory on GPU."""
    logger.info(f"\n💾 GPU MEMORY ALLOCATION TEST")
    
    gpu_device = torch_directml.device()
    
    try:
        # Try to allocate increasingly large tensors
        sizes = [1000, 2000, 3000, 4000, 5000]
        
        for size in sizes:
            logger.info(f"   Allocating {size}x{size} tensor on GPU...")
            tensor = torch.randn(size, size, device=gpu_device)
            memory_mb = (tensor.nbytes / 1024 / 1024)
            logger.info(f"   ✅ Success: {memory_mb:.1f} MB allocated")
            del tensor
        
        logger.info(f"   ✅ GPU memory allocation working properly")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ GPU memory allocation failed: {e}")
        return False

if __name__ == "__main__":
    try:
        logger.info("🔬 DirectML Performance Analysis")
        
        # Test 1: Memory allocation
        memory_ok = test_gpu_memory_allocation()
        
        # Test 2: Performance comparison
        if memory_ok:
            gpu_faster = benchmark_cpu_vs_gpu()
            
            if gpu_faster:
                print("\n✅ CONCLUSION: DirectML is working and accelerating operations!")
                print("🚀 GPU training should provide real speedup")
            else:
                print("\n❌ CONCLUSION: DirectML is NOT providing acceleration")
                print("🐌 Operations are falling back to CPU")
        else:
            print("\n❌ CONCLUSION: GPU memory allocation failed")
            print("🔧 DirectML driver or hardware issue")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ DirectML benchmark failed")