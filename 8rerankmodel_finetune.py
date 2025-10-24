"""
Reranker Model Fine-tuning Script
==================================

WHAT THIS SCRIPT DOES:
---------------------
Fine-tunes a cross-encoder reranker model on your domain-specific data.
Rerankers are used as a second-stage after embedding-based retrieval to
improve precision by scoring query-document pairs.

WHY USE A RERANKER:
------------------
- Embeddings retrieve candidates quickly (top 50-100)
- Reranker scores each candidate precisely (picks best 5-10)
- Combined approach achieves 90%+ accuracy
- Cross-encoder sees both query and document together (more context)

HOW IT WORKS:
------------
1. Loads triplets (anchor/positive/negative) from your training data
2. Creates pairs: (anchor, positive) → score 1.0, (anchor, negative) → score 0.0
3. Fine-tunes a cross-encoder to predict relevance scores
4. Saves the fine-tuned reranker model

USAGE:
------
    python 8rerankmodel_finetune.py                    # Train reranker
    python 8rerankmodel_finetune.py --action test      # Test reranker
    python 8rerankmodel_finetune.py --epochs 5         # Train for 5 epochs

TRAINING DATA:
-------------
Uses the same triplets as your embedding model training.
Expected location: TRAINING_DATA_DIR/{difficulty}/triplets_train.json

MODEL ARCHITECTURE:
------------------
Cross-encoder vs Bi-encoder (embeddings):
- Bi-encoder: Encodes query and document separately, compares vectors
- Cross-encoder: Encodes query + document together, predicts score
- Cross-encoders are more accurate but slower (use after embedding retrieval)

DEPENDENCIES:
------------
- sentence-transformers (for CrossEncoder class)
- torch
- Your training data in triplets format
"""

import json
import torch
import argparse
import os
from pathlib import Path
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# Import config
from scripts.config_embed_training import (
    TRAINING_DATA_DIR,
    RERANKER_BASE_MODEL,
    RERANKER_OUTPUT_PATH,
    BASE_CWD,
    DOCDIR,
    LOG_FILES
)

from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)


def load_triplets(difficulty="hard"):
    """
    Load training triplets from specified difficulty level.
    
    Uses the same triplet data as the embedding model training.
    Each triplet contains: anchor (query), positive (relevant doc), negative (irrelevant doc)
    """
    # Build paths to training and test data files for the specified difficulty
    train_path = TRAINING_DATA_DIR / difficulty / "triplets_train.json"
    test_path = TRAINING_DATA_DIR / difficulty / "triplets_test.json"
    
    # If the specified difficulty doesn't exist, try to find any available difficulty
    if not train_path.exists():
        logger.warning(f"Train file not found: {train_path}")
        logger.info("Trying all difficulty levels...")
        for diff in ["hard", "medium", "easy"]:
            train_path = TRAINING_DATA_DIR / diff / "triplets_train.json"
            test_path = TRAINING_DATA_DIR / diff / "triplets_test.json"
            if train_path.exists():
                logger.info(f"Found training data in: {diff}")
                difficulty = diff
                break
        else:
            # No training data found at all - can't proceed
            raise FileNotFoundError(f"No training data found under {TRAINING_DATA_DIR}")
    
    # Load training triplets from JSON file
    with open(train_path, 'r', encoding='utf-8') as f:
        train_triplets = json.load(f)
    
    # Load test triplets if available (optional but recommended for evaluation)
    test_triplets = []
    if test_path.exists():
        with open(test_path, 'r', encoding='utf-8') as f:
            test_triplets = json.load(f)
    
    logger.info(f"Loaded {len(train_triplets)} training triplets from {difficulty}")
    logger.info(f"Loaded {len(test_triplets)} test triplets")
    
    return train_triplets, test_triplets


def triplets_to_pairs(triplets):
    """
    Convert triplets to training pairs for cross-encoder.
    
    Cross-encoders are trained on (query, document) pairs with relevance scores.
    Each triplet becomes 2 training examples:
    - (anchor, positive) with score 1.0 = "these match, score high"
    - (anchor, negative) with score 0.0 = "these don't match, score low"
    
    This teaches the model to distinguish relevant from irrelevant documents.
    
    Args:
        triplets: List of dicts with 'anchor', 'positive', 'negative' keys
        
    Returns:
        List of InputExample objects for CrossEncoder training (2x the number of triplets)
    """
    examples = []
    
    for triplet in triplets:
        # Create positive pair: anchor + positive document
        # Label 1.0 means "highly relevant"
        examples.append(InputExample(
            texts=[triplet['anchor'], triplet['positive']],
            label=1.0
        ))
        
        # Create negative pair: anchor + negative document  
        # Label 0.0 means "not relevant"
        examples.append(InputExample(
            texts=[triplet['anchor'], triplet['negative']],
            label=0.0
        ))
    
    return examples


def train_reranker(epochs=3, batch_size=16, learning_rate=2e-5, warmup_steps=100):
    """
    Train a cross-encoder reranker model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for learning rate scheduler
    """
    logger.info("="*80)
    logger.info("🚀 RERANKER MODEL FINE-TUNING")
    logger.info("="*80)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load training data
    logger.info("Loading training data...")
    train_triplets, test_triplets = load_triplets()
    
    # Convert triplets to pairs
    train_examples = triplets_to_pairs(train_triplets)
    test_examples = triplets_to_pairs(test_triplets) if test_triplets else []
    
    logger.info(f"Created {len(train_examples)} training pairs ({len(train_triplets)} triplets)")
    if test_examples:
        logger.info(f"Created {len(test_examples)} test pairs ({len(test_triplets)} triplets)")
    
    # Initialize the cross-encoder model
    # If RERANKER_BASE_MODEL is a HuggingFace model ID, it will download automatically
    # If it's a local path, it will load from there
    # num_labels=1 means we're predicting a single relevance score (0.0 to 1.0)
    logger.info(f"Loading base model: {RERANKER_BASE_MODEL}")
    model = CrossEncoder(RERANKER_BASE_MODEL, num_labels=1, device=device)
    
    # Create data loader for batched training
    # Shuffle ensures model doesn't memorize the order of examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Calculate how many training steps we'll do
    steps_per_epoch = len(train_dataloader)  # Number of batches per epoch
    total_steps = steps_per_epoch * epochs    # Total updates to model weights
    
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    # Train the model using the .fit() method
    # Warmup gradually increases learning rate at the start (stabilizes training)
    # show_progress_bar displays a progress bar in the terminal
    logger.info("Starting training...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )
    
    # Save the model
    logger.info(f"Saving model to: {RERANKER_OUTPUT_PATH}")
    RERANKER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(RERANKER_OUTPUT_PATH))
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"Model saved to: {RERANKER_OUTPUT_PATH}")
    
    # Evaluate if we have test data
    if test_examples:
        logger.info("\nRunning evaluation on test set...")
        evaluate_reranker(model, test_examples)


def evaluate_reranker(model=None, test_examples=None):
    """
    Evaluate reranker performance.
    
    Args:
        model: CrossEncoder model (if None, loads from RERANKER_OUTPUT_PATH)
        test_examples: List of test examples (if None, loads from test data)
    """
    logger.info("="*80)
    logger.info("📊 RERANKER EVALUATION")
    logger.info("="*80)
    
    # Load model if not provided
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model from: {RERANKER_OUTPUT_PATH}")
        model = CrossEncoder(str(RERANKER_OUTPUT_PATH), device=device)
    
    # Load test data if not provided
    if test_examples is None:
        _, test_triplets = load_triplets()
        if not test_triplets:
            logger.error("No test data available for evaluation")
            return
        test_examples = triplets_to_pairs(test_triplets)
    
    # Run the model on all test pairs to get predicted relevance scores
    logger.info(f"Evaluating on {len(test_examples)} test pairs...")
    predictions = []
    labels = []
    
    # Get predictions for each test example
    for example in test_examples:
        # model.predict() returns relevance score(s) for the query-document pair
        # [0] gets the first (and only) score since we have num_labels=1
        score = model.predict([example.texts])[0]
        predictions.append(score)
        labels.append(example.label)  # Ground truth: 1.0 for positive, 0.0 for negative
    
    # Convert to numpy arrays for easier calculation
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate accuracy: how often does the positive score higher than the negative?
    # Remember: test_examples are arranged as [pos, neg, pos, neg, ...]
    # So we check pairs: is predictions[0] > predictions[1]? etc.
    accuracy = 0
    total_triplets = len(test_examples) // 2  # Divide by 2 since each triplet = 2 pairs
    
    for i in range(0, len(predictions), 2):
        pos_score = predictions[i]      # Should be high (close to 1.0)
        neg_score = predictions[i + 1]  # Should be low (close to 0.0)
        
        # If positive scored higher than negative, that's a correct ranking
        if pos_score > neg_score:
            accuracy += 1
    
    accuracy = accuracy / total_triplets  # Convert to percentage
    
    # Calculate average scores for positives and negatives
    pos_scores = predictions[::2]   # Every even index (0, 2, 4, ...) is a positive pair
    neg_scores = predictions[1::2]  # Every odd index (1, 3, 5, ...) is a negative pair
    
    mean_pos_score = np.mean(pos_scores)  # Average relevance for positive pairs
    mean_neg_score = np.mean(neg_scores)  # Average relevance for negative pairs
    margin = mean_pos_score - mean_neg_score  # Separation between them (bigger = better)
    
    # Print results
    logger.info("="*80)
    logger.info("📈 RESULTS")
    logger.info("="*80)
    logger.info(f"Accuracy:              {accuracy*100:.2f}% (positive scored higher)")
    logger.info(f"Mean Positive Score:   {mean_pos_score:.4f} (higher is better)")
    logger.info(f"Mean Negative Score:   {mean_neg_score:.4f} (lower is better)")
    logger.info(f"Margin (pos - neg):    {margin:.4f} (higher is better)")
    logger.info("="*80)
    
    return {
        'accuracy': accuracy,
        'mean_pos_score': mean_pos_score,
        'mean_neg_score': mean_neg_score,
        'margin': margin
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune reranker model")
    parser.add_argument("--action", choices=["train", "test"], default="train",
                       help="Action to perform: train or test")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Number of warmup steps (default: 100)")
    
    args = parser.parse_args()
    
    if args.action == "train":
        train_reranker(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps
        )
    elif args.action == "test":
        evaluate_reranker()


if __name__ == "__main__":
    main()
