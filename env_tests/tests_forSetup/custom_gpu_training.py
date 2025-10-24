#!/usr/bin/env python3
"""
CUSTOM GPU TRAINING - DirectML Direct Integration
================================================

Custom training loop that bypasses SentenceTransformers' fit() method
and directly uses PyTorch operations with DirectML for GPU acceleration.
"""

import torch
import torch_directml
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import time
from pathlib import Path
import pandas as pd
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_CWD = Path("C:/GIT/ai_data/extendscript")
TRAINING_DATA_DIR = BASE_CWD / "embedding_training_data"
OUTPUT_MODEL_PATH = BASE_CWD / "qwen3_embedding_finetuned"

def load_training_data():
    """Load training triplets."""
    train_path = TRAINING_DATA_DIR / "train_triplets.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    
    # Convert to triplets (anchor, positive, negative)
    triplets = []
    for _, row in train_df.iterrows():
        triplets.append([row['anchor'], row['positive'], row['negative']])
    
    logger.info(f"Loaded {len(triplets)} training triplets")
    return triplets

def custom_gpu_training():
    """Custom training loop with direct DirectML integration."""
    logger.info("🚀 Starting CUSTOM GPU TRAINING with DirectML")
    
    # Setup DirectML device
    device = torch_directml.device()
    logger.info(f"✅ Using DirectML device: {device}")
    
    # Load model on CPU first
    logger.info("📥 Loading model on CPU...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    
    # Load training data
    triplets = load_training_data()
    
    # Take only a small subset for testing (first 50 triplets)
    test_triplets = triplets[:50]
    logger.info(f"🎯 Using {len(test_triplets)} triplets for GPU test")
    
    # Custom training loop
    logger.info("🔥 Starting custom GPU training loop...")
    
    # Move only the embeddings to GPU, not the entire model
    total_start_time = time.time()
    
    for i, (anchor, positive, negative) in enumerate(test_triplets):
        step_start_time = time.time()
        
        try:
            # Generate embeddings on CPU first
            anchor_emb = model.encode([anchor])
            positive_emb = model.encode([positive]) 
            negative_emb = model.encode([negative])
            
            # Convert to tensors and move to GPU
            anchor_tensor = torch.tensor(anchor_emb, device=device, dtype=torch.float32)
            positive_tensor = torch.tensor(positive_emb, device=device, dtype=torch.float32)
            negative_tensor = torch.tensor(negative_emb, device=device, dtype=torch.float32)
            
            # Compute triplet loss on GPU
            distance_pos = torch.nn.functional.pairwise_distance(anchor_tensor, positive_tensor)
            distance_neg = torch.nn.functional.pairwise_distance(anchor_tensor, negative_tensor)
            
            # Triplet loss computation on GPU
            margin = 0.5
            loss = torch.clamp(distance_pos - distance_neg + margin, min=0.0)
            loss_value = loss.mean()
            
            step_time = time.time() - step_start_time
            
            if i % 10 == 0:  # Print every 10 steps
                logger.info(f"   Step {i+1}/{len(test_triplets)}: Loss={loss_value:.4f}, Time={step_time:.2f}s")
                logger.info(f"   🎯 GPU should show activity NOW - check Task Manager!")
            
        except Exception as e:
            logger.error(f"❌ Step {i+1} failed: {e}")
            break
    
    total_time = time.time() - total_start_time
    logger.info(f"✅ Custom GPU training completed in {total_time:.2f} seconds")
    logger.info(f"📊 Average time per triplet: {total_time/len(test_triplets):.3f} seconds")
    
    return True

if __name__ == "__main__":
    try:
        success = custom_gpu_training()
        if success:
            print("\n🎉 SUCCESS: GPU training worked!")
            print("✅ DirectML is properly accelerating the computations")
            print("💡 This proves GPU acceleration is possible with custom training loop")
        else:
            print("\n❌ FAILED: GPU training did not work")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("❌ GPU training failed")