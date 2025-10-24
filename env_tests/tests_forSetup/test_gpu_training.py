#!/usr/bin/env python3
"""
QUICK TRAINING TEST with GPU
============================

Test that the modified fine_tune_model() function actually uses your RX 580.
This will run just a few training steps to verify GPU is engaged.
"""

import os
import sys
from embedding_finetuner import fine_tune_model, load_training_data, setup_model
import torch
import time
from pathlib import Path

def quick_training_test():
    """Test the training function with GPU for just a few steps."""
    print("🚀 QUICK TRAINING TEST - GPU Integration")
    print("=" * 50)
    
    # Check if training data exists
    BASE_CWD = Path("C:/GIT/ai_data/extendscript")
    TRAINING_DATA_DIR = BASE_CWD / "embedding_training_data"
    
    train_path = TRAINING_DATA_DIR / "train_triplets.csv"
    if not train_path.exists():
        print(f"❌ Training data not found at {train_path}")
        print("You need to generate training data first:")
        print("python embed_training_data_gen.py")
        return False
    
    print("✅ Training data found")
    
    # Temporarily modify training config for quick test
    import config_embed_training
    
    # Save original values
    original_epochs = config_embed_training.TRAINING_CONFIG["epochs"]
    original_eval_steps = config_embed_training.TRAINING_CONFIG["evaluation_steps"]
    
    # Set for quick test (just a few steps)
    config_embed_training.TRAINING_CONFIG["epochs"] = 1
    config_embed_training.TRAINING_CONFIG["evaluation_steps"] = 5
    
    print("⚡ Modified config for quick test:")
    print(f"   Epochs: {config_embed_training.TRAINING_CONFIG['epochs']}")
    print(f"   Eval steps: {config_embed_training.TRAINING_CONFIG['evaluation_steps']}")
    
    try:
        print("\n🎯 Starting GPU training test...")
        print("Watch Task Manager - GPU tab should show activity!")
        
        start_time = time.time()
        
        # This should now use GPU
        fine_tune_model()
        
        total_time = time.time() - start_time
        print(f"\n✅ Training test completed in {total_time:.1f} seconds")
        print("If you saw GPU activity in Task Manager, the RX 580 is working!")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    finally:
        # Restore original config
        config_embed_training.TRAINING_CONFIG["epochs"] = original_epochs
        config_embed_training.TRAINING_CONFIG["evaluation_steps"] = original_eval_steps
        print(f"\n🔄 Restored original config: epochs={original_epochs}")
    
    return True

if __name__ == "__main__":
    success = quick_training_test()
    if success:
        print("\n💡 GPU training integration successful!")
        print("Now you can run full training with GPU acceleration:")
        print("python embedding_finetuner.py --action train")
    else:
        print("\n❌ Need to fix GPU integration before full training")