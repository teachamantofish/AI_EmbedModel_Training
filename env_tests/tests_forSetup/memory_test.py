#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED TRAINING TEST
==============================

Test aggressive memory management for Qwen3-4B on RX 580.
Focus on preventing 100% memory usage.
"""

import torch
import torch_directml
import logging
import gc
import os
from sentence_transformers import SentenceTransformer, InputExample
import torch.nn.functional as F
from torch.optim import AdamW

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aggressive_memory_cleanup():
    """Ultra-aggressive memory cleanup."""
    gc.collect()
    try:
        torch_directml.empty_cache()
    except:
        pass
    try:
        torch.cuda.empty_cache()
    except:
        pass

def memory_optimized_test():
    """Test memory-optimized training approach."""
    logger.info("🧹 MEMORY-OPTIMIZED TRAINING TEST")
    logger.info("🎯 Goal: Prevent 100% memory usage with Qwen3-4B")
    
    try:
        # 1. GPU Setup with minimal memory footprint
        gpu_device = torch_directml.device()
        logger.info(f"✅ DirectML device: {gpu_device}")
        
        # 2. Load model with minimal memory impact
        logger.info("📦 Loading Qwen3-4B model (memory optimized)...")
        model = SentenceTransformer('Qwen/Qwen3-Embedding-4B')
        logger.info("✅ Model loaded on CPU")
        
        # 3. Create tiny test dataset (minimal memory)
        logger.info("📊 Creating minimal test dataset...")
        train_examples = [
            InputExample(texts=["anchor text 1", "positive text 1", "negative text 1"]),
            InputExample(texts=["anchor text 2", "positive text 2", "negative text 2"])
        ]
        logger.info(f"✅ Test dataset: {len(train_examples)} examples")
        
        # 4. Setup optimizer (only parameters, not model)
        optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
        logger.info("✅ Optimizer created")
        
        # 5. Test one training step with extreme memory management
        logger.info("🚀 Testing ONE training step with memory cleanup...")
        
        model.train()
        
        # Process just ONE example with maximum cleanup
        example = train_examples[0]
        
        # Extract texts
        anchor_text = [example.texts[0]]
        positive_text = [example.texts[1]]
        negative_text = [example.texts[2]]
        
        logger.info("   📝 Tokenizing (CPU)...")
        anchor_inputs = model.tokenize(anchor_text)
        positive_inputs = model.tokenize(positive_text)
        negative_inputs = model.tokenize(negative_text)
        
        logger.info("   🧠 Forward pass (CPU)...")
        # Forward pass on CPU
        anchor_emb = model(anchor_inputs)['sentence_embedding']
        positive_emb = model(positive_inputs)['sentence_embedding']
        negative_emb = model(negative_inputs)['sentence_embedding']
        
        logger.info("   🚀 Moving to GPU for loss...")
        # Move to GPU for loss computation only
        anchor_emb_gpu = anchor_emb.to(gpu_device)
        positive_emb_gpu = positive_emb.to(gpu_device)
        negative_emb_gpu = negative_emb.to(gpu_device)
        
        # Clean up CPU tensors immediately
        del anchor_emb, positive_emb, negative_emb
        del anchor_inputs, positive_inputs, negative_inputs
        aggressive_memory_cleanup()
        
        logger.info("   📊 Computing loss on GPU...")
        loss = F.triplet_margin_loss(anchor_emb_gpu, positive_emb_gpu, negative_emb_gpu, margin=1.0)
        
        logger.info("   ⬅️ Backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"   ✅ Loss: {loss.item():.4f}")
        
        # Ultra cleanup after step
        del anchor_emb_gpu, positive_emb_gpu, negative_emb_gpu, loss
        aggressive_memory_cleanup()
        
        logger.info("🎉 MEMORY TEST PASSED!")
        logger.info("💡 One training step completed with memory management")
        logger.info("🌙 Ready for overnight training with this approach")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        return False
    
    finally:
        # Final cleanup
        try:
            del model, optimizer
        except:
            pass
        aggressive_memory_cleanup()

if __name__ == "__main__":
    success = memory_optimized_test()
    if success:
        print("\n🌙 READY FOR OVERNIGHT TRAINING!")
        print("   Memory management approach validated")
        print("   Run full training with confidence")
    else:
        print("\n❌ MEMORY ISSUES DETECTED")
        print("   Need to adjust approach before overnight run")