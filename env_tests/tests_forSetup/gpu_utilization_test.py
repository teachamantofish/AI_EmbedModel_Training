#!/usr/bin/env python3
"""
GPU UTILIZATION OPTIMIZATION TEST
=================================

Test different strategies to maximize GPU utilization:
1. Larger matrices
2. More parallel operations  
3. More intensive computations
"""

import torch
import torch_directml
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_utilization_strategies():
    """Test different approaches to maximize GPU utilization."""
    logger.info("🔥 GPU UTILIZATION OPTIMIZATION TEST")
    
    device = torch_directml.device()
    logger.info(f"✅ Using DirectML device: {device}")
    
    # Strategy 1: Larger matrices
    logger.info("\n🚀 STRATEGY 1: Larger matrices (3000x3000)")
    logger.info("📊 WATCH GPU USAGE - should be higher than 50%!")
    
    start_time = time.time()
    for i in range(50):  # Fewer operations but much larger
        # Much larger matrices
        a = torch.randn(3000, 3000, device=device)
        b = torch.randn(3000, 3000, device=device)
        
        # Intensive operations
        c = torch.matmul(a, b)
        d = torch.relu(c)
        e = torch.sigmoid(d)
        f = torch.tanh(e)
        
        del a, b, c, d, e, f
        
        if (i + 1) % 10 == 0:
            logger.info(f"   Large matrices progress: {i + 1}/50")
            logger.info(f"   🎯 Check GPU usage NOW!")
    
    strategy1_time = time.time() - start_time
    logger.info(f"📊 Strategy 1 time: {strategy1_time:.2f} seconds")
    
    # Strategy 2: Multiple parallel operations
    logger.info("\n🚀 STRATEGY 2: Multiple parallel operations")
    logger.info("📊 WATCH GPU USAGE - attempting to saturate GPU!")
    
    start_time = time.time()
    for i in range(30):
        # Create multiple tensor pairs simultaneously
        tensors_a = [torch.randn(2000, 2000, device=device) for _ in range(4)]
        tensors_b = [torch.randn(2000, 2000, device=device) for _ in range(4)]
        
        # Process all pairs in parallel
        results = []
        for a, b in zip(tensors_a, tensors_b):
            c = torch.matmul(a, b)
            d = torch.relu(c)
            e = torch.sigmoid(d)
            results.append(e)
        
        # Additional cross-operations
        combined = torch.stack(results)
        final = torch.mean(combined, dim=0)
        
        del tensors_a, tensors_b, results, combined, final
        
        if (i + 1) % 10 == 0:
            logger.info(f"   Parallel ops progress: {i + 1}/30")
            logger.info(f"   🎯 Check GPU usage NOW - should be HIGHER!")
    
    strategy2_time = time.time() - start_time
    logger.info(f"📊 Strategy 2 time: {strategy2_time:.2f} seconds")
    
    # Strategy 3: Maximum intensity
    logger.info("\n🚀 STRATEGY 3: Maximum intensity operations")
    logger.info("📊 WATCH GPU USAGE - going for 75%+ utilization!")
    
    start_time = time.time()
    for i in range(20):
        # Very large matrices with complex operations
        a = torch.randn(4000, 4000, device=device)
        b = torch.randn(4000, 4000, device=device)
        
        # Chain of intensive operations
        c = torch.matmul(a, b)
        d = torch.matmul(c, a)  # Another matrix multiplication
        e = torch.relu(d)
        f = torch.sigmoid(e)
        g = torch.tanh(f)
        h = torch.log(torch.abs(g) + 1e-8)  # Logarithm
        result = torch.sum(h)  # Reduction operation
        
        # Force computation to complete
        result.item()
        
        del a, b, c, d, e, f, g, h, result
        
        if (i + 1) % 5 == 0:
            logger.info(f"   Max intensity progress: {i + 1}/20")
            logger.info(f"   🎯 Check GPU usage NOW - aiming for 75%+!")
    
    strategy3_time = time.time() - start_time
    logger.info(f"📊 Strategy 3 time: {strategy3_time:.2f} seconds")
    
    logger.info(f"\n📈 UTILIZATION STRATEGY COMPARISON:")
    logger.info(f"   Strategy 1 (Large matrices): {strategy1_time:.2f}s")
    logger.info(f"   Strategy 2 (Parallel ops): {strategy2_time:.2f}s") 
    logger.info(f"   Strategy 3 (Max intensity): {strategy3_time:.2f}s")
    
    return strategy1_time, strategy2_time, strategy3_time

if __name__ == "__main__":
    try:
        logger.info("🎯 Testing GPU utilization optimization strategies")
        logger.info("📊 Watch Task Manager GPU percentage during each strategy!")
        
        times = test_gpu_utilization_strategies()
        
        print(f"\n🎯 RESULTS - Which strategy achieved highest GPU usage?")
        print(f"Strategy 1 (Large matrices): {times[0]:.2f}s")
        print(f"Strategy 2 (Parallel ops): {times[1]:.2f}s") 
        print(f"Strategy 3 (Max intensity): {times[2]:.2f}s")
        print(f"\n💡 The strategy with highest GPU usage should be fastest!")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ GPU utilization test failed")