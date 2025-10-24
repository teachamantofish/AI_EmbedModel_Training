"""
Multi-Epoch Training Pipeline for Embedding Models
===================================================

WHAT THIS SCRIPT DOES:
---------------------
This script orchestrates a complete training workflow for fine-tuning an embedding model.
It handles everything from combining datasets to running multiple training epochs.

The pipeline follows these steps:
1. COMBINE: Merge training data from multiple sources (e.g., JSX and MIF documentation)
2. SHUFFLE: Randomly mix the combined data for better training
3. TOKENIZE: Convert text into tokens the model can understand
4. TRAIN: Fine-tune the model on your combined dataset
5. SAVE: Store the trained model for later use

WHY USE THIS SCRIPT:
-------------------
- Automatically combines data from multiple directories (configured in DOCLIST)
- Ensures random shuffling for better model generalization
- Handles multi-epoch training (training in stages with progressively harder examples)
- Manages model paths between epochs automatically
- Backs up configuration files before making changes
- Provides detailed logging so you know what's happening

CONFIGURATION:
-------------
Before running, edit scripts/config_embed_training.py to set:
- DOCLIST: Array of directory names containing your training data
  Example: ["extendscript", "mifref"] 
  These are subdirectories under BASE_CWD that contain embedding_training_data folders
  
- BASE_CWD: Base path where your data directories live
  Example: Path("C:/GIT/AI_DataSource/framemaker")
  
- TRAINING_CONFIG: Model settings like batch size, learning rate, LoRA parameters
  
- OUTPUT_MODEL_PATH: Where to save the fine-tuned model

USAGE EXAMPLES:
--------------
Basic usage (single epoch with combined datasets):  python run_multi_epoch_training.py --epochs 1

Multi-epoch training (curriculum learning): python run_multi_epoch_training.py --epochs 3 --start-epoch 1

Skip combining if you already combined the data: python run_multi_epoch_training.py --epochs 1 --skip-combine

See what would happen without actually running: python run_multi_epoch_training.py --epochs 1 --dry-run

Resume from epoch 2 (if epoch 1 already completed): python run_multi_epoch_training.py --epochs 3 --start-epoch 2

COMMAND LINE OPTIONS:
--------------------
--epochs N          : How many training epochs to run (1-3, default: 3)
--start-epoch N     : Which epoch to start from (1-3, default: 1)
--skip-combine      : Don't combine datasets (use existing combined data)
--skip-tokenization : Don't tokenize data (use existing tokenized data)
--dry-run          : Show what would be done without executing

HOW MULTI-EPOCH TRAINING WORKS:
-------------------------------
Epoch 1: Train on your combined dataset with easy negatives
         Model learns basic patterns
         Saves to: OUTPUT_MODEL_PATH

Epoch 2: Load the epoch 1 model, train on medium difficulty negatives
         Model refines its understanding
         Saves to: OUTPUT_MODEL_PATH-epoch2

Epoch 3: Load the epoch 2 model, train on hard negatives
         Model masters difficult distinctions
         Saves to: OUTPUT_MODEL_PATH-epoch3

This progressive difficulty is called "curriculum learning" and often produces
better results than training on all data at once.

DATA STRUCTURE EXPECTED:
-----------------------
Your training data should be organized like this:

C:/GIT/AI_DataSource/framemaker/
├── extendscript/
│   └── embedding_training_data/
│       ├── triplets_train.json  (required)
│       └── triplets_test.json   (optional)
├── mifref/
│   └── embedding_training_data/
│       ├── triplets_train.json  (required)
│       └── triplets_test.json   (optional)

Each triplets JSON file contains an array of training examples:
[
    {
        "anchor": "Query text or question",
        "positive": "Relevant answer or passage",
        "negative": "Irrelevant passage"
    },
    ...
]

WHAT HAPPENS WHEN YOU RUN THIS:
-------------------------------
1. The script reads DOCLIST from config (e.g., ["extendscript", "mifref"])
2. For each directory, it loads triplets_train.json and triplets_test.json
3. All triplets are combined into one large dataset
4. The combined dataset is shuffled with a random seed
5. Combined data is saved to TRAINING_DATA_DIR (first item in DOCLIST by default)
6. The tokenizer converts text to model-readable format
7. The training script fine-tunes the model on your data
8. The trained model is saved to OUTPUT_MODEL_PATH

TROUBLESHOOTING:
---------------
Error: "No training data found in any source"
→ Check that your directories match DOCLIST entries
→ Verify triplets_train.json files exist in each embedding_training_data folder

Error: "Training failed"
→ Check GPU memory (run nvidia-smi)
→ Reduce batch_size in config if out of memory
→ Enable gradient_checkpointing if available

Config changes not taking effect:
→ Script backs up and modifies config during multi-epoch runs
→ Check for timestamped backup files: config_embed_training_backup_*.py
→ The config reflects the last epoch run (this is intentional)

DEPENDENCIES:
------------
This script requires the following to be set up:
- Python virtual environment (.venv)
- PyTorch with CUDA support
- sentence-transformers library
- Your training data prepared as triplets JSON files
- 2tokenize_triplets.py (handles tokenization)
- 4embedmodel_finetuner.py (handles training)

For more details on training configuration, see:
scripts/config_embed_training.py (training parameters, paths, LoRA settings)
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json
import shutil
import random
from datetime import datetime

# Import config (read-only, never modified)
from scripts.config_embed_training import (
    TRAINING_DATA_DIR, 
    OUTPUT_MODEL_PATH, 
    BASE_MODEL,
    TRAINING_CONFIG,
    TOKENIZED_DATA_DIR,
    DOCLIST,
    BASE_CWD,
    CURRENT_EPOCH,
    RUN_EPOCHS
)

from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)

# Configuration for multi-epoch training
# Defines difficulty levels and model checkpointing strategy
TRAINING_STAGES = {
    1: {
        "name": "easy_negatives",
        "difficulty": "easy",
        "description": "Training with easy negatives (baseline)",
        "model_input": BASE_MODEL,  # Start from base model
        "model_output": OUTPUT_MODEL_PATH,
    },
    2: {
        "name": "medium_negatives", 
        "difficulty": "medium",
        "description": "Training with medium negatives (harder examples)",
        "model_input": OUTPUT_MODEL_PATH,  # Load from epoch 1
        "model_output": OUTPUT_MODEL_PATH.parent / f"{OUTPUT_MODEL_PATH.name}-epoch2",
    },
    3: {
        "name": "hard_negatives",
        "difficulty": "hard", 
        "description": "Training with hard negatives (challenging examples)",
        "model_input": OUTPUT_MODEL_PATH.parent / f"{OUTPUT_MODEL_PATH.name}-epoch2",  # Load from epoch 2
        "model_output": OUTPUT_MODEL_PATH.parent / f"{OUTPUT_MODEL_PATH.name}-epoch3",
    }
}


def activate_venv():
    """
    Ensure virtual environment is activated for subprocess calls.
    
    WHAT IT DOES:
    - Checks if a .venv directory exists with a Python executable
    - Returns the path to the virtual environment's Python
    - Falls back to system Python if no venv found
    
    WHY IT'S NEEDED:
    - Subprocesses don't automatically use the virtual environment
    - We need to explicitly point to the venv's Python executable
    - Ensures all dependencies are available when calling other scripts
    
    RETURNS:
    - String path to Python executable (either venv or system)
    """
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_command(cmd, description):
    """
    Run a shell command and handle errors gracefully.
    
    WHAT IT DOES:
    - Executes a command in a subprocess
    - Prints what it's doing for visibility
    - Captures and reports any errors
    - Returns success/failure status
    
    PARAMETERS:
    - cmd: List of command parts (e.g., ["python", "script.py"])
    - description: Human-readable description for logging
    
    RETURNS:
    - True if command succeeded
    - False if command failed
    
    EXAMPLE:
    run_command(["python", "train.py"], "Training the model")
    """
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False


def load_triplets(json_path):
    """
    Load training triplets from a JSON file.
    
    WHAT IT DOES:
    - Opens a JSON file containing training examples
    - Parses the JSON data into a Python list
    - Handles missing files gracefully (returns empty list)
    
    PARAMETERS:
    - json_path: Path to the JSON file to load
    
    RETURNS:
    - List of triplet dictionaries, each containing:
      {
          "anchor": "The query or question text",
          "positive": "A relevant answer or passage",
          "negative": "An irrelevant passage"
      }
    - Empty list [] if file not found
    
    EXAMPLE USAGE:
    triplets = load_triplets(Path("data/triplets_train.json"))
    print(f"Loaded {len(triplets)} training examples")
    """
    if not json_path.exists():
        print(f"⚠️  Warning: File not found: {json_path}")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_triplets(triplets, json_path):
    """
    Save training triplets to a JSON file.
    
    WHAT IT DOES:
    - Creates the output directory if it doesn't exist
    - Writes triplets to a JSON file with nice formatting
    - Uses UTF-8 encoding to preserve special characters
    
    PARAMETERS:
    - triplets: List of triplet dictionaries to save
    - json_path: Where to save the JSON file
    
    JSON FORMAT:
    The output is formatted with:
    - indent=2: Nice readable formatting with 2-space indents
    - ensure_ascii=False: Preserves Unicode characters (emojis, accents, etc.)
    
    EXAMPLE USAGE:
    save_triplets(combined_data, Path("output/triplets_train.json"))
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)


def combine_datasets():
    """
    Combine training data from all sources listed in DOCLIST for each difficulty level.
    
    WHAT IT DOES:
    1. Loops through each difficulty level (easy, medium, hard)
    2. For each difficulty, loops through each directory in DOCLIST (e.g., "extendscript", "mifref")
    3. Loads triplets_train.json and triplets_test.json from each source's difficulty subdirectory
    4. Combines all training triplets for that difficulty into one large list
    5. Combines all test triplets for that difficulty into one large list
    6. Shuffles both lists with a time-based random seed
    7. Saves the combined, shuffled data to TRAINING_DATA_DIR/difficulty/
    
    WHY THIS IS IMPORTANT:
    - Training on diverse data sources improves model generalization
    - Shuffling prevents the model from learning source-specific biases
    - Different shuffle each run (time-based seed) helps avoid overfitting patterns
    - Combined data means one training run covers all your domains at each difficulty level
    
    DOCLIST EXAMPLE:
    If DOCLIST = ["extendscript", "mifref"], this function loads from:
    - C:/GIT/AI_DataSource/framemaker/extendscript/embedding_training_data/easy/triplets_train.json
    - C:/GIT/AI_DataSource/framemaker/mifref/embedding_training_data/easy/triplets_train.json
    
    And combines them into:
    - C:/GIT/AI_DataSource/framemaker/extendscript/embedding_training_data/easy/triplets_train.json
      (overwrites the first source's easy directory with combined easy data)
    
    RETURNS:
    - True if successful (found training data and saved combined versions for all difficulties)
    - False if no training data found in any source for any difficulty
    
    RANDOM SEED:
    Uses current timestamp as seed, so each run shuffles differently.
    This is good for training - prevents memorizing a specific order.
    """
    print("\n" + "="*80)
    print("📦 COMBINING TRAINING DATASETS")
    print("="*80)
    
    difficulties = ['easy', 'medium', 'hard']
    overall_success = False
    
    # Shuffle seed for consistent shuffling across all difficulties
    seed = int(datetime.now().timestamp())
    print(f"\n🎲 Using shuffle seed: {seed}")
    
    # Process each difficulty level
    for difficulty in difficulties:
        print(f"\n{'='*80}")
        print(f"📊 Processing difficulty: {difficulty.upper()}")
        print(f"{'='*80}")
        
        all_train_triplets = []
        all_test_triplets = []
        
        # Loop through each data source directory
        for doc_source in DOCLIST:
            data_dir = BASE_CWD / doc_source / "embedding_training_data" / difficulty
            
            print(f"\n📂 Loading from: {doc_source}/{difficulty}")
            print(f"   Path: {data_dir}")
            
            # Load training triplets
            train_path = data_dir / "triplets_train.json"
            train_triplets = load_triplets(train_path)
            if train_triplets:
                print(f"   ✅ Loaded {len(train_triplets)} training triplets")
                all_train_triplets.extend(train_triplets)
            else:
                print(f"   ⚠️  No training data found")
            
            # Load test triplets (optional)
            test_path = data_dir / "triplets_test.json"
            test_triplets = load_triplets(test_path)
            if test_triplets:
                print(f"   ✅ Loaded {len(test_triplets)} test triplets")
                all_test_triplets.extend(test_triplets)
            else:
                print(f"   ⚠️  No test data found")
        
        # Check if we got any training data for this difficulty
        if not all_train_triplets:
            print(f"\n⚠️  Warning: No training data found for {difficulty} difficulty")
            continue
        
        overall_success = True
        
        print(f"\n📊 Total combined for {difficulty}:")
        print(f"   Training: {len(all_train_triplets)} triplets")
        print(f"   Test: {len(all_test_triplets)} triplets")
        
        # Shuffle with consistent seed across difficulties
        print(f"   🔀 Shuffling...")
        random.seed(seed)
        random.shuffle(all_train_triplets)
        if all_test_triplets:
            random.shuffle(all_test_triplets)
        
        # Save to combined directory for this difficulty
        combined_dir = TRAINING_DATA_DIR / difficulty
        print(f"\n💾 Saving combined {difficulty} data to: {combined_dir}")
        
        save_triplets(all_train_triplets, combined_dir / "triplets_train.json")
        print(f"   ✅ Saved triplets_train.json ({len(all_train_triplets)} triplets)")
        
        if all_test_triplets:
            save_triplets(all_test_triplets, combined_dir / "triplets_test.json")
            print(f"   ✅ Saved triplets_test.json ({len(all_test_triplets)} triplets)")
    
    # Check if we found any data at all
    if not overall_success:
        print("\n❌ Error: No training data found in any source for any difficulty!")
        return False
    
    print(f"\n{'='*80}")
    print("✅ Dataset combining completed successfully!")
    print(f"{'='*80}")
    return True


def check_training_data_exists(epoch_num, stage_config):
    """
    Verify that training data files exist for this epoch.
    
    WHAT IT DOES:
    - Checks if triplets_train.json exists in the expected directory
    - Checks if triplets_test.json exists (optional but recommended)
    - Prints helpful error messages if files are missing
    
    PARAMETERS:
    - epoch_num: Current epoch number (for error messages)
    - stage_config: Dictionary with data directory info for this epoch
    
    RETURNS:
    - True if training data found
    - False if training data missing
    
    WHY THIS CHECK:
    - Prevents wasting time starting training only to fail later
    - Gives clear guidance on what's missing and where to put it
    - Helps catch configuration errors early
    """
    data_dir = TRAINING_DATA_DIR / stage_config["difficulty"]
    
    # Check for triplets JSON files
    train_file = data_dir / "triplets_train.json"
    test_file = data_dir / "triplets_test.json"
    
    if not train_file.exists() or not test_file.exists():
        print(f"⚠️  Warning: Training data not found for epoch {epoch_num}")
        print(f"   Expected: {train_file}")
        print(f"   Expected: {test_file}")
        print(f"\n   Please run create_training_data.py to create training data with different difficulty levels:")
        print(f"   1. Run your data generation script for '{stage_config['name']}'")
        print(f"   2. Save triplets to: {data_dir}/")
        return False
    
    print(f"✅ Training data found for epoch {epoch_num}: {data_dir}")
    return True


def run_tokenization(epoch_num, stage_config):
    """
    Run the tokenization script to convert text into tokens.
    
    WHAT IT DOES:
    - Calls 2tokenize_triplets.py to tokenize your training data
    - Converts human-readable text into numeric tokens the model understands
    - Saves tokenized data to TOKENIZED_DATA_DIR
    
    WHY TOKENIZATION IS NEEDED:
    - Models don't understand raw text, only numbers (tokens)
    - Each word/subword gets converted to a token ID
    - Tokenization happens once, then training uses the tokenized data
    
    PARAMETERS:
    - epoch_num: Current epoch number (for logging)
    - stage_config: Dictionary with info about this epoch
    
    RETURNS:
    - True if tokenization succeeded
    - False if tokenization failed
    
    EXAMPLE:
    Text: "Hello world" 
    → Tokens: [15496, 995]
    → These numbers are what the model actually processes
    """
    python_exe = activate_venv()
    
    description = f"Tokenizing data for epoch {epoch_num} ({stage_config['name']})"
    cmd = [python_exe, "2tokenize_triplets.py"]
    
    return run_command(cmd, description)


def run_training(epoch_num, stage_config):
    """
    Run the training script to fine-tune the model.
    
    WHAT IT DOES:
    - Calls 4embedmodel_finetuner.py to train the model
    - Loads tokenized data and the base/previous model
    - Runs training loop for specified number of epochs
    - Saves the fine-tuned model
    
    THIS IS THE CORE TRAINING STEP:
    - Model learns to create better embeddings for your domain
    - Updates model weights based on your training data
    - Uses triplet loss to learn: positive pairs should be close,
      negative pairs should be far apart
    
    PARAMETERS:
    - epoch_num: Current epoch number (for logging)
    - stage_config: Dictionary with info about this epoch
    
    RETURNS:
    - True if training succeeded
    - False if training failed
    
    TRAINING PROGRESSION:
    Epoch 1: Base model + your data → Epoch 1 model
    Epoch 2: Epoch 1 model + harder data → Epoch 2 model (better)
    Epoch 3: Epoch 2 model + hardest data → Epoch 3 model (best)
    """
    python_exe = activate_venv()
    
    description = f"Training epoch {epoch_num} - {stage_config['description']}"
    cmd = [python_exe, "4embedmodel_finetuner.py"]
    
    return run_command(cmd, description)


def main():
    parser = argparse.ArgumentParser(description="Multi-epoch training pipeline")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of epochs to run (1-3)")
    parser.add_argument("--start-epoch", type=int, default=1, help="Starting epoch (1-3)")
    parser.add_argument("--skip-tokenization", action="store_true", help="Skip tokenization step")
    parser.add_argument("--skip-combine", action="store_true", help="Skip dataset combining step")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    if args.epochs < 1 or args.epochs > 3:
        print("❌ Error: epochs must be between 1 and 3")
        sys.exit(1)
    
    if args.start_epoch < 1 or args.start_epoch > args.epochs:
        print(f"❌ Error: start-epoch must be between 1 and {args.epochs}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🎯 MULTI-EPOCH TRAINING PIPELINE")
    print("="*80)
    print(f"Data sources: {', '.join(DOCLIST)}")
    print(f"Total epochs: {args.epochs}")
    print(f"Starting from epoch: {args.start_epoch}")
    print(f"Skip combining: {args.skip_combine}")
    print(f"Skip tokenization: {args.skip_tokenization}")
    print(f"Dry run: {args.dry_run}")
    print("="*80 + "\n")
    
    # Store original config for restoration
    config_backup = None
    
    try:
        # Step 0: Combine datasets from all sources in DOCLIST
        if not args.skip_combine and not args.dry_run:
            if not combine_datasets():
                print("\n❌ Failed to combine datasets")
                sys.exit(1)
        elif args.skip_combine:
            print("⏭️  Skipping dataset combining (using existing combined data)")
        elif args.dry_run:
            print("[DRY RUN] Would combine datasets from:", ', '.join(DOCLIST))
        
        for epoch in range(args.start_epoch, args.epochs + 1):
            stage_config = TRAINING_STAGES[epoch]
            
            print(f"\n{'#'*80}")
            print(f"# EPOCH {epoch}/{args.epochs}: {stage_config['name'].upper()}")
            print(f"# {stage_config['description']}")
            print(f"{'#'*80}\n")
            
            # Check if training data exists
            if not check_training_data_exists(epoch, stage_config):
                print(f"\n⚠️  Skipping epoch {epoch} - training data not found")
                print("   Create the training data first, then re-run this script")
                continue
            
            if args.dry_run:
                print(f"[DRY RUN] Would tokenize data: {not args.skip_tokenization}")
                print(f"[DRY RUN] Would train epoch {epoch}")
                continue
            
            # Tokenize data if needed
            if not args.skip_tokenization:
                if not run_tokenization(epoch, stage_config):
                    print(f"❌ Tokenization failed for epoch {epoch}")
                    break
            else:
                print(f"⏭️  Skipping tokenization for epoch {epoch}")
            
            # Run training
            if not run_training(epoch, stage_config):
                print(f"❌ Training failed for epoch {epoch}")
                break
            
            print(f"\n✅ Epoch {epoch} completed successfully!")
            print(f"   Model saved to: {stage_config['model_output']}\n")
        
        print("\n" + "="*80)
        print("🎉 TRAINING PIPELINE COMPLETED!")
        print("="*80)
        print(f"\nFinal model location: {TRAINING_STAGES[args.epochs]['model_output']}")
        print("\nNext steps:")
        print("1. Test the model: python 4embedmodel_finetuner.py --action test")
        print("2. Export for use: python 4embedmodel_finetuner.py --action export")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        raise
    finally:
        print("\n💡 Note: To resume or run a different epoch, update CURRENT_EPOCH in config_embed_training.py")


if __name__ == "__main__":
    main()
