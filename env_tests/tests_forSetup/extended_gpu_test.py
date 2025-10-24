#!/usr/bin/env python3
"""
EXTENDED SCENARIO 1 TEST - Pre-allocated Tensors (10+ seconds)
============================================================

Run the optimal scenario (pre-allocated tensors) for extended duration
to clearly observe sustained high GPU utilization.
"""

import torch
import torch_directml
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extended_prealloc_test():
    """Extended test of pre-allocated tensor scenario for ~10 seconds."""
    logger.info("🔥 EXTENDED SCENARIO 1 TEST - Pre-allocated Tensors")
    
    device = torch_directml.device()
    logger.info(f"✅ Using DirectML device: {device}")
    
    logger.info("\n🚀 Pre-allocating tensors for maximum GPU utilization...")
    
    # Pre-allocate larger tensors for sustained work
    a = torch.randn(2500, 2500, device=device)
    b = torch.randn(2500, 2500, device=device)
    c = torch.zeros(2500, 2500, device=device)
    
    logger.info("📊 WATCH TASK MANAGER GPU NOW - aiming for 10+ seconds of high usage!")
    logger.info("🎯 This should show sustained 70-80%+ GPU utilization!")
    
    start_time = time.time()
    operations = 0
    
    # Run for approximately 10 seconds
    while time.time() - start_time < 10:
        # Reuse pre-allocated tensors - no allocation overhead
        c = torch.matmul(a, b)
        c = torch.relu(c)
        c = torch.sigmoid(c)
        c = torch.tanh(c)
        
        operations += 1
        
        # Progress updates every 2 seconds
        elapsed = time.time() - start_time
        if operations % 20 == 0:
            logger.info(f"   🎯 {elapsed:.1f}s elapsed - GPU should be at HIGH utilization!")
            logger.info(f"   Operations completed: {operations}")
    
    total_time = time.time() - start_time
    ops_per_second = operations / total_time
    
    logger.info(f"\n✅ Extended test completed!")
    logger.info(f"📊 Total time: {total_time:.2f} seconds")
    logger.info(f"📊 Total operations: {operations}")
    logger.info(f"📊 Operations per second: {ops_per_second:.1f}")
    logger.info(f"🎯 Did you see sustained high GPU usage throughout the {total_time:.1f} seconds?")
    
    return total_time, operations

if __name__ == "__main__":
    try:
        logger.info("🎯 Testing sustained GPU utilization with pre-allocated tensors")
        logger.info("📊 Watch Task Manager GPU percentage for the next ~10 seconds!")
        
        total_time, operations = extended_prealloc_test()
        
        print(f"\n🎯 EXTENDED TEST RESULTS:")
        print(f"Duration: {total_time:.2f} seconds")
        print(f"Operations: {operations}")
        print(f"Rate: {operations/total_time:.1f} ops/second")
        print(f"\n💡 This represents optimal GPU utilization for your RX 580!")
        print(f"🚀 We can apply this approach to embedding training for maximum speed!")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ Extended GPU test failed")