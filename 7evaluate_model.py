"""
Evaluate Trained Embedding Models
==================================

Simple script to test and compare embedding models (epoch 1, 2, 3, or base model).

WHAT THIS DOES:
- Loads a trained embedding model
- Runs evaluation on test data
- Calculates similarity scores for anchor-positive vs anchor-negative pairs
- Reports accuracy: how often positive is closer than negative

USAGE:
    python 7evaluate_model.py --epoch 3          # Test epoch 3 model
    python 7evaluate_model.py --epoch 2          # Test epoch 2 model
    python 7evaluate_model.py --epoch 1          # Test epoch 1 model
    python 7evaluate_model.py --epoch base       # Test original base model
    python 7evaluate_model.py --all              # Test all models and compare

WHAT IT REPORTS:
- Mean similarity between anchors and positives (should be high, close to 1.0)
- Mean similarity between anchors and negatives (should be low, close to 0.0)
- Accuracy: % of cases where positive is more similar than negative
- Margin: Average difference between positive and negative similarities
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Import config
from scripts.config_embed_training import (
    BASE_CWD,
    OUTPUT_MODEL_PATH,
    BASE_MODEL,
    TRAINING_DATA_DIR,
    DOCDIR,
    TRAINING_DATA_DIR
)

from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=TRAINING_DATA_DIR, log_level='INFO', headers=LOG_HEADER)


def get_model_path(epoch):
    """Get the path to a model based on epoch number or 'base'."""
    if epoch == "base":
        return BASE_MODEL
    elif epoch == 1:
        return str(OUTPUT_MODEL_PATH)
    elif epoch == 2:
        return str(OUTPUT_MODEL_PATH.parent / f"{OUTPUT_MODEL_PATH.name}-epoch2")
    elif epoch == 3:
        return str(OUTPUT_MODEL_PATH.parent / f"{OUTPUT_MODEL_PATH.name}-epoch3")
    else:
        raise ValueError(f"Invalid epoch: {epoch}. Must be 1, 2, 3, or 'base'")


def load_test_data(difficulty="hard"):
    """Load test triplets from the specified difficulty level."""
    test_path = TRAINING_DATA_DIR / difficulty / "triplets_test.json"
    
    if not test_path.exists():
        print(f"⚠️  Test file not found: {test_path}")
        print(f"   Trying all difficulty levels...")
        
        # Try all difficulties
        for diff in ["hard", "medium", "easy"]:
            test_path = TRAINING_DATA_DIR / diff / "triplets_test.json"
            if test_path.exists():
                print(f"   ✅ Found test data in: {diff}")
                difficulty = diff
                break
        else:
            raise FileNotFoundError(f"No test data found in any difficulty level under {TRAINING_DATA_DIR}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    
    print(f"📊 Loaded {len(triplets)} test triplets from {difficulty} difficulty")
    return triplets


def evaluate_model(model_path, triplets):
    """
    Evaluate an embedding model on test triplets.
    
    Returns:
        dict: Evaluation metrics including accuracy, mean similarities, and margin
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {Path(model_path).name}")
    print(f"{'='*80}")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer(model_path, device=device)
    
    # Extract texts
    anchors = [t['anchor'] for t in triplets]
    positives = [t['positive'] for t in triplets]
    negatives = [t['negative'] for t in triplets]
    
    # Generate embeddings with larger batch size for GPU efficiency
    batch_size = 32 if device == "cuda" else 8
    print(f"Generating embeddings for {len(triplets)} triplets (batch_size={batch_size})...")
    anchor_embs = model.encode(anchors, convert_to_tensor=True, show_progress_bar=True, 
                               batch_size=batch_size, device=device)
    positive_embs = model.encode(positives, convert_to_tensor=True, show_progress_bar=True, 
                                 batch_size=batch_size, device=device)
    negative_embs = model.encode(negatives, convert_to_tensor=True, show_progress_bar=True, 
                                 batch_size=batch_size, device=device)
    
    # Convert to numpy for similarity calculation
    anchor_embs = anchor_embs.cpu().numpy()
    positive_embs = positive_embs.cpu().numpy()
    negative_embs = negative_embs.cpu().numpy()
    
    # Calculate cosine similarities
    pos_similarities = np.array([
        cosine_similarity([anchor_embs[i]], [positive_embs[i]])[0][0]
        for i in range(len(triplets))
    ])
    
    neg_similarities = np.array([
        cosine_similarity([anchor_embs[i]], [negative_embs[i]])[0][0]
        for i in range(len(triplets))
    ])
    
    # Calculate metrics
    accuracy = np.mean(pos_similarities > neg_similarities)
    mean_pos_sim = np.mean(pos_similarities)
    mean_neg_sim = np.mean(neg_similarities)
    margin = np.mean(pos_similarities - neg_similarities)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📈 RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:               {accuracy*100:.2f}% (positive closer than negative)")
    print(f"Mean Positive Similarity: {mean_pos_sim:.4f} (higher is better)")
    print(f"Mean Negative Similarity: {mean_neg_sim:.4f} (lower is better)")
    print(f"Margin (pos - neg):      {margin:.4f} (higher is better)")
    print(f"{'='*80}\n")
    
    return {
        'model': Path(model_path).name,
        'accuracy': accuracy,
        'mean_pos_sim': mean_pos_sim,
        'mean_neg_sim': mean_neg_sim,
        'margin': margin
    }


def compare_models(results):
    """Print a comparison table of multiple model results."""
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'Accuracy':>10} {'Margin':>10}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['model']:<40} {r['accuracy']*100:>9.2f}% {r['margin']:>10.4f}")
    
    print(f"{'='*80}\n")
    
    # Find best model
    best = max(results, key=lambda x: x['accuracy'])
    print(f"🏆 Best model: {best['model']} ({best['accuracy']*100:.2f}% accuracy)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained embedding models")
    parser.add_argument("--epoch", type=str, default="3", 
                       help="Which epoch to test: 1, 2, 3, or 'base' (default: 3)")
    parser.add_argument("--all", action="store_true",
                       help="Test all models (base, epoch1, epoch2, epoch3) and compare")
    parser.add_argument("--difficulty", type=str, default="hard",
                       choices=["easy", "medium", "hard"],
                       help="Which difficulty test set to use (default: hard)")
    
    args = parser.parse_args()
    
    # Load test data
    triplets = load_test_data(args.difficulty)
    
    if args.all:
        # Evaluate all models
        results = []
        for epoch in ["base", 1, 2, 3]:
            try:
                model_path = get_model_path(epoch)
                if not Path(model_path).exists():
                    print(f"⚠️  Skipping {epoch} - model not found at {model_path}")
                    continue
                result = evaluate_model(model_path, triplets)
                results.append(result)
            except Exception as e:
                print(f"❌ Error evaluating epoch {epoch}: {e}")
        
        if len(results) > 1:
            compare_models(results)
    else:
        # Evaluate single model
        epoch = args.epoch if args.epoch == "base" else int(args.epoch)
        model_path = get_model_path(epoch)
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        evaluate_model(model_path, triplets)


if __name__ == "__main__":
    main()
