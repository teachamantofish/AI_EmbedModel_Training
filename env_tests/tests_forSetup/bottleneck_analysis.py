#!/usr/bin/env python3
"""
GPU BOTTLENECK ANALYSIS
======================

Identify what's limiting GPU utilization by testing different scenarios:
1. Memory-bound vs compute-bound operations
2. Allocation patterns
3. Operation complexity
"""

import torch
import torch_directml
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bottleneck_scenarios():
    """Test different scenarios to identify GPU utilization bottleneck."""
    logger.info("🔍 GPU BOTTLENECK ANALYSIS")
    
    device = torch_directml.device()
    logger.info(f"✅ Using DirectML device: {device}")
    
    # Scenario 1: Pre-allocated tensors (eliminate allocation overhead)
    logger.info("\n🚀 SCENARIO 1: Pre-allocated tensors (eliminate allocation overhead)")
    logger.info("📊 WATCH GPU USAGE - should be higher without allocation overhead!")
    
    # Pre-allocate tensors
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)
    c = torch.zeros(2000, 2000, device=device)
    
    start_time = time.time()
    for i in range(100):
        # Reuse pre-allocated tensors - DirectML compatible
        c = torch.matmul(a, b)
        c = torch.relu(c)
        c = torch.sigmoid(c)
        
        if (i + 1) % 25 == 0:
            logger.info(f"   Pre-allocated progress: {i + 1}/100")
            logger.info(f"   🎯 Check GPU usage - no allocation overhead!")
    
    scenario1_time = time.time() - start_time
    logger.info(f"📊 Scenario 1 time: {scenario1_time:.2f} seconds")
    
    # Scenario 2: Smaller operations, higher frequency
    logger.info("\n🚀 SCENARIO 2: Smaller, high-frequency operations")
    logger.info("📊 WATCH GPU USAGE - testing different operation size!")
    
    start_time = time.time()
    for i in range(500):  # More operations, smaller size
        a_small = torch.randn(1000, 1000, device=device)
        b_small = torch.randn(1000, 1000, device=device)
        
        c_small = torch.matmul(a_small, b_small)
        d_small = torch.relu(c_small)
        
        del a_small, b_small, c_small, d_small
        
        if (i + 1) % 100 == 0:
            logger.info(f"   Small ops progress: {i + 1}/500")
            logger.info(f"   🎯 Check GPU usage - smaller operations!")
    
    scenario2_time = time.time() - start_time
    logger.info(f"📊 Scenario 2 time: {scenario2_time:.2f} seconds")
    
    # Scenario 3: Simple operations only (minimal DirectML overhead)
    logger.info("\n🚀 SCENARIO 3: Simple operations only (minimal overhead)")
    logger.info("📊 WATCH GPU USAGE - pure matrix multiplication!")
    
    start_time = time.time()
    for i in range(200):
        a_simple = torch.randn(2000, 2000, device=device)
        b_simple = torch.randn(2000, 2000, device=device)
        
        # Only matrix multiplication - simplest GPU operation
        c_simple = torch.matmul(a_simple, b_simple)
        
        del a_simple, b_simple, c_simple
        
        if (i + 1) % 50 == 0:
            logger.info(f"   Simple ops progress: {i + 1}/200")
            logger.info(f"   🎯 Check GPU usage - pure matmul only!")
    
    scenario3_time = time.time() - start_time
    logger.info(f"📊 Scenario 3 time: {scenario3_time:.2f} seconds")
    
    # Scenario 4: Batch operations (process multiple at once)
    logger.info("\n🚀 SCENARIO 4: Batch operations (process multiple simultaneously)")
    logger.info("📊 WATCH GPU USAGE - batch processing!")
    
    start_time = time.time()
    for i in range(50):
        # Create batch of matrices
        batch_a = torch.randn(8, 1500, 1500, device=device)  # Batch of 8 matrices
        batch_b = torch.randn(8, 1500, 1500, device=device)
        
        # Batch matrix multiplication
        batch_c = torch.matmul(batch_a, batch_b)
        batch_d = torch.relu(batch_c)
        
        del batch_a, batch_b, batch_c, batch_d
        
        if (i + 1) % 10 == 0:
            logger.info(f"   Batch ops progress: {i + 1}/50")
            logger.info(f"   🎯 Check GPU usage - batch processing!")
    
    scenario4_time = time.time() - start_time
    logger.info(f"📊 Scenario 4 time: {scenario4_time:.2f} seconds")
    
    logger.info(f"\n📈 BOTTLENECK ANALYSIS RESULTS:")
    logger.info(f"   Scenario 1 (Pre-allocated): {scenario1_time:.2f}s")
    logger.info(f"   Scenario 2 (Small/frequent): {scenario2_time:.2f}s")
    logger.info(f"   Scenario 3 (Simple ops): {scenario3_time:.2f}s")
    logger.info(f"   Scenario 4 (Batch ops): {scenario4_time:.2f}s")
    logger.info(f"\n💡 The scenario with highest GPU usage indicates the optimal approach!")
    
    return scenario1_time, scenario2_time, scenario3_time, scenario4_time

if __name__ == "__main__":
    try:
        logger.info("🔬 Analyzing GPU utilization bottlenecks")
        logger.info("📊 Watch GPU percentage - which scenario achieves highest usage?")
        
        times = test_bottleneck_scenarios()
        
        fastest_scenario = min(enumerate(times, 1), key=lambda x: x[1])
        
        print(f"\n🎯 BOTTLENECK ANALYSIS COMPLETE")
        print(f"Scenario 1 (Pre-allocated): {times[0]:.2f}s")
        print(f"Scenario 2 (Small/frequent): {times[1]:.2f}s")
        print(f"Scenario 3 (Simple ops): {times[2]:.2f}s")
        print(f"Scenario 4 (Batch ops): {times[3]:.2f}s")
        print(f"\n🏆 Fastest: Scenario {fastest_scenario[0]} ({fastest_scenario[1]:.2f}s)")
        print(f"💡 This scenario likely achieved highest GPU utilization!")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ Bottleneck analysis failed")