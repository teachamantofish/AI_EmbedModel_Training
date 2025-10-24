#!/usr/bin/env python3
"""
SIMPLE CPU-ONLY TRAINING TEST
============================

Since SentenceTransformers + DirectML are incompatible, let's test
CPU-only training to make sure the training pipeline works.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_finetuner import load_training_data, setup_model, create_evaluator
from config_embed_training import *
import torch
from sentence_transformers import losses
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_cpu_training():
    """Test training with CPU-only, no DirectML complications."""
    logger.info("🚀 SIMPLE CPU-ONLY TRAINING TEST")
    logger.info("Testing the training pipeline without GPU complications")
    
    # Force CPU device
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Load data and model
    train_examples, test_examples = load_training_data()
    model = setup_model(device=device)
    
    # Use very small subset for quick test
    mini_train = train_examples[:10]
    mini_test = test_examples[:5]
    
    logger.info(f"Testing with {len(mini_train)} train examples, {len(mini_test)} test examples")
    
    # Create data loader with small batch
    train_dataloader = DataLoader(mini_train, shuffle=True, batch_size=2)
    
    # Define loss function
    train_loss = losses.TripletLoss(model=model)
    
    # Create evaluator
    evaluator = create_evaluator(mini_test)
    
    logger.info("Starting mini training session...")
    
    # Simple training configuration
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,
        evaluation_steps=2,
        warmup_steps=1,
        optimizer_params={'lr': 5e-6},
        output_path=str(Path("test_cpu_model")),
        save_best_model=False,
        show_progress_bar=True,
    )
    
    logger.info("✅ CPU training test completed successfully!")
    logger.info("The training pipeline works - just without GPU acceleration")

if __name__ == "__main__":
    simple_cpu_training()