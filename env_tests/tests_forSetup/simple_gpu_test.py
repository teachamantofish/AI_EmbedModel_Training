#!/usr/bin/env python3
"""
SIMPLE GPU TRAINING TEST - Direct Tensor Operations
==================================================

Simplified test that directly creates trainable tensors on GPU
to ensure we can see GPU activity in Task Manager.
"""

import torch
import torch_directml
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_gpu_training_test():
    """Simple GPU training test with direct tensor operations."""
    logger.info("🔥 SIMPLE GPU TRAINING TEST - Direct Tensor Operations")
    
    # Setup DirectML device
    device = torch_directml.device()
    logger.info(f"✅ Using DirectML device: {device}")
    
    # Create trainable parameters directly on GPU
    logger.info("🚀 Creating trainable tensors directly on GPU...")
    
    # Simulate embedding dimensions (384 for MiniLM)
    embedding_dim = 384
    batch_size = 10
    
    # Create trainable "embedding" tensors on GPU
    anchor_embeddings = torch.randn(batch_size, embedding_dim, device=device, requires_grad=True)
    positive_embeddings = torch.randn(batch_size, embedding_dim, device=device, requires_grad=True)
    negative_embeddings = torch.randn(batch_size, embedding_dim, device=device, requires_grad=True)
    
    logger.info(f"✅ Created tensors on GPU: {anchor_embeddings.shape}")
    logger.info(f"✅ Tensors require gradients: {anchor_embeddings.requires_grad}")
    
    # Create optimizer
    optimizer = torch.optim.Adam([anchor_embeddings, positive_embeddings, negative_embeddings], lr=0.001)
    logger.info("✅ Optimizer created for GPU tensors")
    
    logger.info("\n🔥 Starting intensive GPU training loop...")
    logger.info("📊 WATCH TASK MANAGER GPU NOW - should show HIGH activity!")
    
    total_start_time = time.time()
    
    for epoch in range(5):  # 5 epochs for sustained activity
        logger.info(f"\n🚀 EPOCH {epoch + 1}/5 - GPU should be VERY active NOW!")
        
        epoch_loss = 0.0
        
        for step in range(20):  # 20 steps per epoch
            step_start_time = time.time()
            
            # 1. FORWARD PASS - Compute triplet loss on GPU
            # Distance computations (GPU intensive)
            distance_pos = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings)
            distance_neg = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings)
            
            # Triplet loss with margin
            margin = 0.5
            loss = torch.clamp(distance_pos - distance_neg + margin, min=0.0).mean()
            
            # 2. BACKWARD PASS - Compute gradients on GPU
            optimizer.zero_grad()
            loss.backward()
            
            # 3. WEIGHT UPDATES - Update tensors on GPU
            optimizer.step()
            
            # Extra GPU-intensive operations to ensure visibility
            for _ in range(3):
                large_tensor = torch.randn(1000, 1000, device=device)
                result = torch.matmul(large_tensor, large_tensor)
                result = torch.relu(result)
                result = torch.sigmoid(result)
                del large_tensor, result
            
            epoch_loss += loss.item()
            step_time = time.time() - step_start_time
            
            if step % 5 == 0:  # Print every 5 steps
                logger.info(f"     Step {step+1}/20: Loss={loss.item():.4f}, Time={step_time:.3f}s")
                logger.info(f"     🎯 GPU SHOULD BE VERY ACTIVE - Check Task Manager!")
        
        avg_loss = epoch_loss / 20
        logger.info(f"📊 Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    
    total_time = time.time() - total_start_time
    logger.info(f"\n✅ GPU training completed in {total_time:.2f} seconds")
    logger.info(f"📊 Total operations: {5 * 20} forward+backward passes with extra GPU work")
    
    return True

if __name__ == "__main__":
    try:
        success = simple_gpu_training_test()
        if success:
            print("\n🎉 SUCCESS: GPU training with direct tensors worked!")
            print("✅ Forward pass, backward pass, and weight updates on GPU")
            print("✅ DirectML should have shown HIGH GPU activity")
            print("💡 If you saw GPU activity, DirectML acceleration is working!")
        else:
            print("\n❌ FAILED: GPU training did not work")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ GPU training failed")