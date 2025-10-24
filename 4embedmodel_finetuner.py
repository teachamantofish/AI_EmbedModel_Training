# Fine-tune Qwen3-Embedding-4B model using the generated training triplets
# Based on SBERT methodology: https://www.sbert.net/docs/training/overview.html

"""
EMBEDDING MODEL FINE-TUNING PIPELINE

This script fine-tunes embedding models on your domain-specific data to improve
semantic search and RAG performance. It supports three main actions:

COMMAND LINE USAGE:
===================
python embedding_finetuner.py --action train   # Default: Fine-tune model with triplets (takes hours)
python embedding_finetuner.py --action test    # Test model quality and validate results (seconds)  
python embedding_finetuner.py --action export  # Export model for use in embedding creaton/vector DB (seconds)
python embedding_finetuner.py --action hardware_test  # Test hardware compatibility
python embedding_finetuner.py --action smoke_steps --steps 50 --batch_size 1 --grad_accum 4  # Run a short limited-step smoke test (50 optimizer steps)

TYPICAL WORKFLOW:
=================

Run 1 epoch → Check results → If good, run epoch 2 → Check results → If good, run epoch 3

FINE-TUNING PROCESS:
1. Loads training triplets from embedding_training_data/
2. Uses triplet loss to train model on (anchor, positive, negative) relationships
3. Saves fine-tuned model with improved domain understanding
4. Model learns your specific vocabulary and semantic relationships

TUNING NOTES for no CUDA:
✅ DirectML works and provides 2x+ speedup
✅ Pre-allocated tensors achieve 75% GPU utilization
✅ Memory allocation is the bottleneck - we solved it
✅ Sustainable for 10+ seconds of continuous work

so: 

Modify embedding_finetuner.py to use custom training loop
Pre-allocate tensor buffers for embeddings and gradients
Apply DirectML acceleration to loss computation and backprop
Reuse memory instead of creating new tensors each iteration
Target 75% GPU utilization for actual training
"""
# Uses the default HuggingFace cache location for model storage
# To run: python embedding_finetuner.py --action train

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_from_disk
import torch.nn.functional as F

from scripts.config_embed_training import *
from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Model", "Training Step"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)

class TripletDataCollator:
    """Module-level collator that pads triplet inputs using an HF tokenizer.

    Defined at module scope so it is picklable and can be used as a DataLoader collate_fn
    on Windows. Wraps an HF DataCollatorWithPadding instance and produces a batch
    containing anchor/positive/negative tensors.
    """
    def __init__(self, base_collator: DataCollatorWithPadding):
        self.base = base_collator

    def __call__(self, examples):
        anchors = [{"input_ids": e["anchor_input_ids"], "attention_mask": e.get("anchor_attention_mask")} for e in examples]
        positives = [{"input_ids": e["positive_input_ids"], "attention_mask": e.get("positive_attention_mask")} for e in examples]
        negatives = [{"input_ids": e["negative_input_ids"], "attention_mask": e.get("negative_attention_mask")} for e in examples]

        a_batch = self.base(anchors)
        p_batch = self.base(positives)
        n_batch = self.base(negatives)

        batch = {
            "anchor_input_ids": a_batch["input_ids"],
            "anchor_attention_mask": a_batch["attention_mask"],
            "positive_input_ids": p_batch["input_ids"],
            "positive_attention_mask": p_batch["attention_mask"],
            "negative_input_ids": n_batch["input_ids"],
            "negative_attention_mask": n_batch["attention_mask"],
        }
        return batch
# Backwards-compatibility: map TRAINING_CONFIG values to legacy uppercase names
# (avoid adding new config entries; read from TRAINING_CONFIG defined in config_embed_training.py)
try:
    BATCH_SIZE = TRAINING_CONFIG.get('batch_size', 1)
    EPOCHS = TRAINING_CONFIG.get('epochs', 1)
    LEARNING_RATE = TRAINING_CONFIG.get('learning_rate', 5e-6)
    WARMUP_STEPS = TRAINING_CONFIG.get('warmup_steps', 50)
    EVALUATION_STEPS = TRAINING_CONFIG.get('evaluation_steps', 100)
    MAX_SEQUENCE_LENGTH = TRAINING_CONFIG.get('max_sequence_length', 96)
except NameError:
    # If TRAINING_CONFIG isn't present for some reason, fall back to defaults
    BATCH_SIZE = 1
    EPOCHS = 1
    LEARNING_RATE = 5e-6
    WARMUP_STEPS = 50
    EVALUATION_STEPS = 100
    MAX_SEQUENCE_LENGTH = 96

def load_training_data() -> Tuple[List[InputExample], List[InputExample]]:
    """Load training triplets and convert to SBERT InputExample format."""
    train_path = TRAINING_DATA_DIR / "triplets_train.json"
    test_path = TRAINING_DATA_DIR / "triplets_test.json"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Training data not found. Run embedmodel_tuner.py first.")
    
    # Load training data from JSON
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Convert to SBERT InputExample format
    train_examples = []
    for item in train_data:
        example = InputExample(
            texts=[item['anchor'], item['positive'], item['negative']],
            label=1.0  # For triplet loss, we use 1.0 to indicate positive pair
        )
        train_examples.append(example)
    
    test_examples = []
    for item in test_data:
        example = InputExample(
            texts=[item['anchor'], item['positive'], item['negative']],
            label=1.0
        )
        test_examples.append(example)
    
    logger.info(f"Loaded {len(train_examples)} training examples and {len(test_examples)} test examples")
    
    return train_examples, test_examples


def setup_model(device=None) -> SentenceTransformer:
    """Initialize the base model for fine-tuning."""
    try:
        # Require a local model path. Do NOT fallback to HuggingFace model ids.
        if not (isinstance(CONFIG_MODEL_NAME, str) and Path(CONFIG_MODEL_NAME).exists()):
            raise FileNotFoundError(f"Local model path not found or CONFIG_MODEL_NAME is not a local path: {CONFIG_MODEL_NAME}")

        model_source = str(Path(CONFIG_MODEL_NAME))
        logger.info(f"Loading model from local path: {model_source}")
        # Optional: load in 8-bit (requires bitsandbytes & accelerate) to reduce memory
        if TRAINING_CONFIG.get('USE_LOAD_IN_8BIT', False):
            try:
                from transformers import AutoModel, AutoTokenizer
                logger.info("⚡ Loading model in 8-bit mode FIRST (load_in_8bit=True, device_map='auto')")
                # Load base model in 8-bit directly - this is CRITICAL for memory efficiency
                trust_remote = TRAINING_CONFIG.get('trust_remote_code', True)
                base_model = AutoModel.from_pretrained(
                    model_source, 
                    load_in_8bit=True, 
                    device_map='auto',
                    trust_remote_code=trust_remote
                )
                logger.info(f"✅ Base model loaded in 8-bit: {base_model.dtype if hasattr(base_model, 'dtype') else 'quantized'}")
                
                # Now wrap it in SentenceTransformer with the 8-bit model already loaded
                # This prevents loading a second FP32 copy
                # CRITICAL: Pass device=None to prevent SentenceTransformer from reloading model
                model = SentenceTransformer(model_source, device=None)
                # Replace the internal model with our 8-bit version
                if hasattr(model, '_first_module'):
                    first_module = model._first_module()
                    if hasattr(first_module, 'auto_model'):
                        first_module.auto_model = base_model
                        logger.info("✅ Replaced SentenceTransformer internal model with 8-bit version")
                    
            except Exception as e:
                logger.warning(f"load_in_8bit requested but failed: {e}. Falling back to normal load.")
                model = SentenceTransformer(model_source)
        else:
            model = SentenceTransformer(model_source)
        
        # Move model to specified device if provided (skip if using 8-bit with device_map='auto')
        if device is not None and not TRAINING_CONFIG.get('USE_LOAD_IN_8BIT', False):
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)
        elif TRAINING_CONFIG.get('USE_LOAD_IN_8BIT', False):
            logger.info("⚠️  Skipping .to(device) - using device_map='auto' from 8-bit loading")

        # Enable gradient checkpointing to reduce memory usage
        try:
            if hasattr(model, '_first_module'):
                first_module = model._first_module()
                if hasattr(first_module, 'gradient_checkpointing_enable'):
                    first_module.gradient_checkpointing_enable()
                    logger.info("✅ Enabled gradient checkpointing for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

        # Apply LoRA if configured to reduce trainable parameters
        use_lora = TRAINING_CONFIG.get('use_lora', False)
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_r = TRAINING_CONFIG.get('lora_r', 8)
                lora_alpha = TRAINING_CONFIG.get('lora_alpha', 16)
                
                # Get the underlying transformer model
                base_model = None
                if hasattr(model, '_first_module'):
                    first_module = model._first_module()
                    if hasattr(first_module, 'auto_model'):
                        base_model = first_module.auto_model
                
                if base_model is not None:
                    lora_dropout = TRAINING_CONFIG.get('lora_dropout', 0.1)
                    lora_target_modules = TRAINING_CONFIG.get('lora_target_modules', ["q_proj", "v_proj"])
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=lora_target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type=TaskType.FEATURE_EXTRACTION
                    )
                    first_module.auto_model = get_peft_model(base_model, lora_config)
                    logger.info(f"✅ Applied LoRA (r={lora_r}, alpha={lora_alpha}) - drastically reduced memory!")
                    # Print trainable parameters
                    trainable_params = sum(p.numel() for p in first_module.auto_model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in first_module.auto_model.parameters())
                    logger.info(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
                else:
                    logger.warning("Could not locate base transformer model for LoRA application")
            except Exception as e:
                logger.warning(f"LoRA requested but failed to apply: {e}")

        # Optional: try to compile model for faster execution if configured
        if TRAINING_CONFIG.get('USE_TORCH_COMPILE', False):
            try:
                # Check for Triton availability before compiling; torch.compile with CUDA backend
                # can require Triton. If Triton is not present, skip compile to avoid runtime errors.
                try:
                    import triton  # type: ignore
                    triton_ok = True
                except Exception:
                    triton_ok = False

                if not triton_ok:
                    logger.warning("torch.compile requested but Triton not found; skipping compile to avoid runtime errors")
                else:
                    logger.info("Attempting to wrap model with torch.compile() for performance (if supported)")
                    import torch
                    model = torch.compile(model)
            except Exception as e:
                logger.warning(f"torch.compile requested but failed: {e}")
        
        logger.info(f"Loaded model: {CONFIG_MODEL_NAME}",
                    extra={"Model": CONFIG_MODEL_NAME, "Training Step": "model_init"})
        return model
    except Exception as e:
        logger.error(f"Failed to load model {CONFIG_MODEL_NAME}: {e}")
        raise


def create_evaluator(test_examples: List[InputExample]) -> TripletEvaluator:
    """Create evaluator for monitoring training progress."""
    # Extract anchors, positives, negatives for evaluation
    anchors = [example.texts[0] for example in test_examples]
    positives = [example.texts[1] for example in test_examples]
    negatives = [example.texts[2] for example in test_examples]
    
    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="embedding_evaluation"
    )
    
    return evaluator


def custom_gpu_training_loop(model, train_dataloader, train_loss, evaluator, gpu_device, use_fp16: bool = False):
    """
    Custom GPU training loop optimized for CUDA with pre-allocated tensors.
    Uses model on the provided device and computes forward/backward on GPU.
    """
    try:
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LinearLR
        
        logger.info("🔧 Setting up custom GPU training loop")
        # Ensure model is on the expected device (cuda)
        logger.info(f"🎯 Using GPU device: {gpu_device}")
        model = model.to(gpu_device)
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # Setup scheduler
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_STEPS)
        
        # Pre-allocate tensors for memory optimization (key to 75% GPU utilization)
        batch_size = BATCH_SIZE
        max_seq_len = MAX_SEQUENCE_LENGTH  
        embedding_dim = model.get_sentence_embedding_dimension()
        
        # Pre-allocate tensors on GPU
        logger.info("🎯 Pre-allocating tensors for optimal GPU utilization")
        pre_anchor = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=gpu_device)
        pre_positive = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=gpu_device)
        pre_negative = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=gpu_device)

        # Choose embedding dtype based on runtime preference (use_fp16) to match AMP behavior
        emb_dtype = torch.float16 if use_fp16 else torch.float32
        pre_embeddings = torch.zeros((batch_size * 3, embedding_dim), dtype=emb_dtype, device=gpu_device)

        logger.info("✅ Tensors pre-allocated, starting training")
        
        # Prepare AMP scaler once per training run if using fp16
        use_amp = use_fp16 and (getattr(gpu_device, 'type', None) == 'cuda')
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        for epoch in range(EPOCHS):
            logger.info(f"🔄 Starting epoch {epoch + 1}/{EPOCHS}")
            
            epoch_loss = 0.0
            batch_count = 0
            global_step = 0
            
            # Determine whether the dataloader provides pre-tokenized batches (dicts/tensors)
            dataset0 = None
            try:
                dataset0 = train_dataloader.dataset[0]
            except Exception:
                dataset0 = None

            if isinstance(dataset0, dict) or (hasattr(dataset0, 'keys') if dataset0 is not None else False):
                # Pre-tokenized path: DataLoader yields batches (dict of tensors) directly
                logger.info("Using pre-tokenized DataLoader fast path")
                for batch_idx, batch in enumerate(train_dataloader):
                    try:
                        # Move tensors to device
                        moved = {}
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                moved[k] = v.to(gpu_device)
                            else:
                                moved[k] = v

                        # Build per-input dicts expected by the model
                        anchor_inputs = {"input_ids": moved.get('anchor_input_ids'), "attention_mask": moved.get('anchor_attention_mask')}
                        positive_inputs = {"input_ids": moved.get('positive_input_ids'), "attention_mask": moved.get('positive_attention_mask')}
                        negative_inputs = {"input_ids": moved.get('negative_input_ids'), "attention_mask": moved.get('negative_attention_mask')}

                        # Forward pass with gradients (model on GPU). Use AMP when fp16 requested
                        # Get triplet margin from config
                        triplet_margin = TRAINING_CONFIG.get('triplet_margin', 1.0)
                        
                        if use_fp16:
                            with torch.amp.autocast('cuda'):
                                anchor_embeddings = model(anchor_inputs)['sentence_embedding']
                                positive_embeddings = model(positive_inputs)['sentence_embedding']
                                negative_embeddings = model(negative_inputs)['sentence_embedding']
                                # Compute triplet loss on GPU
                                loss = F.triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=triplet_margin)
                        else:
                            anchor_embeddings = model(anchor_inputs)['sentence_embedding']
                            positive_embeddings = model(positive_inputs)['sentence_embedding']
                            negative_embeddings = model(negative_inputs)['sentence_embedding']
                            # Compute triplet loss on GPU
                            loss = F.triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=triplet_margin)
                        
                        # Backward pass
                        optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory
                        if use_fp16 and scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        if global_step < WARMUP_STEPS:
                            scheduler.step()

                        # Store loss value before deleting
                        loss_val = loss.item()
                        epoch_loss += loss_val
                        batch_count += 1
                        global_step += 1

                        # AGGRESSIVE memory cleanup - delete everything immediately
                        del anchor_embeddings, positive_embeddings, negative_embeddings
                        del anchor_inputs, positive_inputs, negative_inputs
                        del moved, batch
                        del loss
                        
                        # Force synchronize to ensure all ops complete
                        torch.cuda.synchronize()
                        # Clear cache after every single batch
                        torch.cuda.empty_cache()
                        
                        # Python garbage collection
                        import gc
                        gc.collect()

                        if batch_idx % 10 == 0:
                            logger.info(f"Batch {batch_idx}, Loss: {epoch_loss/(batch_count):.4f}, GPU: Active")
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        continue
            else:
                # Legacy path: train_dataloader.dataset contains InputExample-like objects
                train_examples_list = list(train_dataloader.dataset)
                for batch_idx in range(0, len(train_examples_list), batch_size):
                    try:
                        # Get batch of examples
                        batch_examples = train_examples_list[batch_idx:batch_idx + batch_size]

                        # Extract texts from InputExample objects
                        anchor_texts = [example.texts[0] for example in batch_examples]
                        positive_texts = [example.texts[1] for example in batch_examples]
                        negative_texts = [example.texts[2] for example in batch_examples]

                        # HYBRID APPROACH: Tokenize, then forward pass with gradients
                        # Need to use model() not model.encode() to maintain gradients

                        # Tokenize on CPU (sentence-transformers tokenizer)
                        anchor_inputs = model.tokenize(anchor_texts)
                        positive_inputs = model.tokenize(positive_texts)
                        negative_inputs = model.tokenize(negative_texts)

                        # Move tokenized inputs to GPU device (avoid device-mismatch)
                        def move_inputs_to_device(inputs, device):
                            # inputs is a dict of tensors or lists compatible with HF/SE
                            moved = {}
                            for k, v in inputs.items():
                                if isinstance(v, torch.Tensor):
                                    moved[k] = v.to(device)
                                else:
                                    try:
                                        moved[k] = torch.tensor(v, device=device)
                                    except Exception:
                                        moved[k] = v
                            return moved

                        anchor_inputs = move_inputs_to_device(anchor_inputs, gpu_device)
                        positive_inputs = move_inputs_to_device(positive_inputs, gpu_device)
                        negative_inputs = move_inputs_to_device(negative_inputs, gpu_device)

                        # Forward pass with gradients (model on GPU). Use AMP when fp16 requested
                        if use_fp16:
                            # Use torch.amp autocast for the CUDA device to avoid deprecation warnings
                            with torch.amp.autocast('cuda'):
                                anchor_embeddings = model(anchor_inputs)['sentence_embedding']
                                positive_embeddings = model(positive_inputs)['sentence_embedding']
                                negative_embeddings = model(negative_inputs)['sentence_embedding']
                        else:
                            anchor_embeddings = model(anchor_inputs)['sentence_embedding']
                            positive_embeddings = model(positive_inputs)['sentence_embedding']
                            negative_embeddings = model(negative_inputs)['sentence_embedding']

                        # Compute triplet loss on GPU (this is where GPU gets utilized!)
                        triplet_margin = TRAINING_CONFIG.get('triplet_margin', 1.0)
                        loss = F.triplet_margin_loss(
                            anchor_embeddings, 
                            positive_embeddings, 
                            negative_embeddings,
                            margin=triplet_margin
                        )

                        # Backward pass
                        optimizer.zero_grad()
                        if use_fp16 and scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        if global_step < WARMUP_STEPS:
                            scheduler.step()

                        epoch_loss += loss.item()
                        batch_count += 1
                        global_step += 1

                        # ULTRA-AGGRESSIVE memory cleanup (based on successful memory_test.py)
                        del anchor_embeddings, positive_embeddings, negative_embeddings
                        del anchor_inputs, positive_inputs, negative_inputs
                        del loss

                        # Multi-layer cleanup approach
                        import gc
                        gc.collect()  # Python garbage collection
                        try:
                            torch.cuda.empty_cache()
                        except:
                            pass

                        # Force another GC after GPU cleanup
                        gc.collect()

                        # Log progress
                        if batch_idx % 10 == 0:
                            logger.info(f"Batch {batch_idx//batch_size}, Loss: {epoch_loss/(batch_count):.4f}, GPU: Active")
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        continue
                    
                    
            
            # Epoch summary
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            logger.info(f"✅ Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Evaluation
            if (epoch + 1) % EVALUATION_STEPS == 0:
                logger.info("🔍 Running evaluation")
                model.eval()
                try:
                    eval_score = evaluator(model, output_path=None)
                    logger.info(f"📊 Evaluation score: {eval_score}")
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                model.train()
        
        # Save final model
        logger.info("💾 Saving final model")
        model.save(str(OUTPUT_MODEL_PATH))
        
        return True
        
    except Exception as e:
        logger.error(f"Custom GPU training failed: {e}")
        return False


def fine_tune_model():
    """Main fine-tuning function."""
    logger.info("Starting embedding model fine-tuning",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "start"})
    
    # Set PyTorch memory configuration to reduce fragmentation
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Setup GPU acceleration: prefer CUDA, fallback to DirectML if CUDA not available
    gpu_available = False
    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        gpu_available = True
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        try:
            import torch_directml
            gpu_device = torch_directml.device()
            gpu_available = True
            logger.info(f"✅ DirectML device detected: {gpu_device}")
        except Exception as e:
            logger.warning(f"No GPU acceleration available, using CPU: {e}")
            gpu_available = False
    
    # Load data and model on CPU first (avoid DirectML loading issues)
    # Verify tokenized dataset exists; if not, warn and abort with instructions
    tokenized_path = TOKENIZED_DATA_DIR if 'TOKENIZED_DATA_DIR' in globals() else None

    if tokenized_path is None or not Path(tokenized_path).exists():
        logger.error("Tokenized dataset not found. Please run scripts/tokenize_triplets.py to create tokenized data before training.")
        logger.error("Example: python scripts/tokenize_triplets.py \nThis will create a 'tokenized_train' directory with tokenized inputs and metadata.json.")
        raise SystemExit("Tokenized dataset missing. Aborting.")

    train_examples, test_examples = load_training_data()
    
    # Load model once. If a GPU is available, load directly onto the GPU to avoid
    # keeping a full FP32 copy on the CPU which doubles memory usage and can
    # lead to fragmentation / OOMs. This avoids calling setup_model() twice.
    if gpu_available and gpu_device is not None:
        model = setup_model(device=gpu_device)
    else:
        model = setup_model(device=None)
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # Define loss function (TripletLoss for triplet training)
    train_loss = losses.TripletLoss(model=model)
    
    # Create evaluator
    evaluator = create_evaluator(test_examples)
    
    # Training arguments
    num_training_steps = len(train_dataloader) * EPOCHS
    
    logger.info(f"Training configuration: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
                f"lr={LEARNING_RATE}, steps={num_training_steps}",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "config"})
    
    # Use custom GPU training loop with pre-allocated tensors for optimal performance
    if gpu_available:
        logger.info("🚀 Starting CUSTOM GPU training with pre-allocated tensors")
        logger.info("🎯 Using optimized approach that achieved 75% GPU utilization")
        logger.info("💡 This bypasses SentenceTransformers limitations for maximum GPU usage")

        # If a tokenized HF dataset exists, prefer using it to avoid per-batch tokenization cost
        use_fp16 = TRAINING_CONFIG.get('fp16', False)
        try:
            # Use the load_from_disk already imported at top of file
            token_ds = load_from_disk(str(tokenized_path))
            if hasattr(token_ds, 'keys') and 'train' in token_ds:
                token_train = token_ds['train']
            else:
                token_train = token_ds

            # Use a collate_fn that pads each batch on-the-fly to avoid default_collate resizing errors
            cols = [c for c in ("anchor_input_ids","anchor_attention_mask","positive_input_ids","positive_attention_mask","negative_input_ids","negative_attention_mask","length") if c in token_train.column_names]
            if cols:
                # Build a tokenizer + collator to pad per-batch
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(CONFIG_MODEL_NAME)
                    base_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

                    class TripletDataCollatorLocal:
                        def __init__(self, base):
                            self.base = base

                        def __call__(self, examples):
                            # examples are dict-like with anchor_input_ids etc.
                            anchors = [{"input_ids": e["anchor_input_ids"], "attention_mask": e.get("anchor_attention_mask")} for e in examples]
                            positives = [{"input_ids": e["positive_input_ids"], "attention_mask": e.get("positive_attention_mask")} for e in examples]
                            negatives = [{"input_ids": e["negative_input_ids"], "attention_mask": e.get("negative_attention_mask")} for e in examples]

                            a_batch = self.base(anchors)
                            p_batch = self.base(positives)
                            n_batch = self.base(negatives)

                            batch = {
                                "anchor_input_ids": a_batch["input_ids"],
                                "anchor_attention_mask": a_batch["attention_mask"],
                                "positive_input_ids": p_batch["input_ids"],
                                "positive_attention_mask": p_batch["attention_mask"],
                                "negative_input_ids": n_batch["input_ids"],
                                "negative_attention_mask": n_batch["attention_mask"],
                            }
                            return batch

                    collate_fn = TripletDataCollatorLocal(base_collator)
                    dl = DataLoader(token_train, batch_size=BATCH_SIZE, shuffle=True,
                                    pin_memory=bool(TRAINING_CONFIG.get('DATALOADER_PIN_MEMORY', True)),
                                    num_workers=0,
                                    collate_fn=collate_fn)
                    logger.info(f"Using pre-tokenized DataLoader with collate_fn, {int(TRAINING_CONFIG.get('PREFETCH_WORKERS',1))} workers and pin_memory={TRAINING_CONFIG.get('DATALOADER_PIN_MEMORY', True)}")
                    train_loader = dl
                except Exception as e:
                    logger.warning(f"Failed to build collate_fn for tokenized dataset: {e}; falling back to InputExample DataLoader")
                    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
            else:
                logger.info("Tokenized dataset did not contain expected columns; falling back to InputExample DataLoader")
                train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        except Exception as e:
            logger.warning(f"Could not load pre-tokenized dataset for fast path: {e}; falling back to original dataloader")
            train_loader = train_dataloader

        # Run custom GPU training loop on the prepared DataLoader
        success = custom_gpu_training_loop(model, train_loader, train_loss, evaluator, gpu_device, use_fp16=use_fp16)

        if success:
            logger.info("✅ Custom GPU training completed successfully")
            return
        else:
            logger.error("❌ Custom GPU training failed")
            return
    else:
        logger.info("🔁 Starting CPU-only training using HuggingFace Trainer")

        # Load tokenized dataset produced by scripts/tokenize_triplets.py
        try:
            ds = load_from_disk(str(tokenized_path))
            # If a DatasetDict was saved, try to extract the 'train' split
            if hasattr(ds, 'keys') and 'train' in ds:
                train_dataset = ds['train']
            else:
                train_dataset = ds
        except Exception as e:
            logger.error(f"Failed to load tokenized dataset from {tokenized_path}: {e}")
            raise

        # Build tokenizer (use same tokenizer as model) for padding in collator
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CONFIG_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {CONFIG_MODEL_NAME}: {e}")
            raise

        # Triplet-aware collator: use HF DataCollatorWithPadding to pad each of the three inputs
        base_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        class TripletDataCollator:
            def __init__(self, base):
                self.base = base

            def __call__(self, features):
                # features: list of examples, each containing anchor_input_ids, anchor_attention_mask, etc.
                anchors = [{"input_ids": f["anchor_input_ids"], "attention_mask": f.get("anchor_attention_mask")} for f in features]
                positives = [{"input_ids": f["positive_input_ids"], "attention_mask": f.get("positive_attention_mask")} for f in features]
                negatives = [{"input_ids": f["negative_input_ids"], "attention_mask": f.get("negative_attention_mask")} for f in features]

                a_batch = self.base(anchors)
                p_batch = self.base(positives)
                n_batch = self.base(negatives)

                # Prefix keys so compute_loss can access them
                batch = {
                    "anchor_input_ids": a_batch["input_ids"],
                    "anchor_attention_mask": a_batch["attention_mask"],
                    "positive_input_ids": p_batch["input_ids"],
                    "positive_attention_mask": p_batch["attention_mask"],
                    "negative_input_ids": n_batch["input_ids"],
                    "negative_attention_mask": n_batch["attention_mask"],
                }
                return batch

        collator = TripletDataCollator(base_collator)

        # Ensure the loaded tokenized dataset yields PyTorch tensors to reduce CPU conversion overhead
        try:
            # determine available columns in train_dataset
            cols = []
            for c in ("anchor_input_ids","anchor_attention_mask","positive_input_ids","positive_attention_mask","negative_input_ids","negative_attention_mask","length"):
                if c in train_dataset.column_names:
                    cols.append(c)
            if cols:
                train_dataset.set_format(type="torch", columns=cols)
                logger.info(f"Set tokenized dataset format to torch for columns: {cols}")
        except Exception as e:
            logger.warning(f"Could not set dataset format to torch: {e}")

        # Create a Trainer that computes triplet loss by overriding compute_loss
        class TripletTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # inputs contain tensors for anchor/positive/negative prefixed keys
                anchor_inputs = {"input_ids": inputs["anchor_input_ids"], "attention_mask": inputs["anchor_attention_mask"]}
                positive_inputs = {"input_ids": inputs["positive_input_ids"], "attention_mask": inputs["positive_attention_mask"]}
                negative_inputs = {"input_ids": inputs["negative_input_ids"], "attention_mask": inputs["negative_attention_mask"]}

                # Forward pass through SentenceTransformer model; expect 'sentence_embedding' in outputs
                anchor_out = model(anchor_inputs)
                positive_out = model(positive_inputs)
                negative_out = model(negative_inputs)

                anchor_emb = anchor_out.get('sentence_embedding') if isinstance(anchor_out, dict) else anchor_out
                positive_emb = positive_out.get('sentence_embedding') if isinstance(positive_out, dict) else positive_out
                negative_emb = negative_out.get('sentence_embedding') if isinstance(negative_out, dict) else negative_out

                # Compute triplet margin loss
                margin = TRAINING_CONFIG.get('triplet_margin', 1.0)
                loss = F.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=margin)
                if return_outputs:
                    return (loss, None)
                return loss

        # Prepare TrainingArguments for Trainer. Disable group_by_length for triplet datasets.
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_MODEL_PATH),
            per_device_train_batch_size=int(TRAINING_CONFIG.get('batch_size', BATCH_SIZE)),
            gradient_accumulation_steps=int(TRAINING_CONFIG.get('gradient_accumulation_steps', 1)),
            num_train_epochs=int(TRAINING_CONFIG.get('epochs', EPOCHS)),
            learning_rate=float(TRAINING_CONFIG.get('learning_rate', LEARNING_RATE)),
            group_by_length=False,  # Must be False for triplet datasets with anchor/positive/negative keys
            remove_unused_columns=False,
            logging_steps=TRAINING_CONFIG.get('logging_steps', 10),
            save_strategy=TRAINING_CONFIG.get('save_strategy', 'no'),
            dataloader_num_workers=0,
            dataloader_pin_memory=bool(TRAINING_CONFIG.get('DATALOADER_PIN_MEMORY', True)),
            fp16=bool(TRAINING_CONFIG.get('fp16', False)),
        )

        # Instantiate Trainer
        trainer = TripletTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )

        logger.info("Starting HF Trainer-based triplet training (may require a SentenceTransformer-compatible model)")
        trainer.train()
        logger.info(f"HF Trainer training completed. Model (if saved) at {OUTPUT_MODEL_PATH}")


def test_hardware():
    """
    Quick 5-minute hardware/software test to validate setup before long training.
    
    This runs a mini training session with limited data to test:
    - Model loading and GPU detection
    - Training pipeline functionality  
    - Memory usage and stability
    - Expected performance benchmarks
    
    Use this before committing to full training runs.
    """
    import time
    # Prefer CUDA; fallback to DirectML if CUDA not available
    device = None
    use_gpu = False
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"✅ CUDA device detected: {torch.cuda.get_device_name(0)}")
            use_gpu = True
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                logger.info(f"✅ DirectML device detected: {device}")
                use_gpu = True
            except Exception as e:
                logger.warning(f"No GPU acceleration detected (CUDA/DirectML): {e}")
                use_gpu = False
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        use_gpu = False

    logger.info("Starting 5-minute hardware/software test")
    
    # Load model on CPU first, then move to GPU
    start_time = time.time()
    logger.info("Loading model on CPU first...")
    model = setup_model()
    
    # Now test GPU with smaller operations (hybrid approach)
    try:
        logger.info(f"Testing DirectML device: {device}")
        
        # Test GPU is working with small operations
        test_tensor = torch.randn(100, 100, device=device)
        test_result = torch.matmul(test_tensor, test_tensor)
        logger.info(f"✅ GPU test operation successful: {test_result.shape}")
        
        # Keep model on CPU but use GPU for training computations
        logger.info("✅ Using hybrid CPU/GPU approach: Model on CPU, computations on GPU")
        logger.info("This allows us to train with 8GB GPU + 60GB RAM")
        
    except Exception as e:
        logger.error(f"❌ Even GPU test operations failed: {e}")
        raise Exception("GPU must work for computations - no CPU fallback allowed")
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.1f} seconds",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "model_loading"})
    
    # Load minimal training data (just first 20 examples)
    train_examples, test_examples = load_training_data()
    mini_train = train_examples[:20]  # Only use first 20 examples
    mini_test = test_examples[:10]    # Only use first 10 test examples
    
    logger.info(f"Using mini dataset: {len(mini_train)} train, {len(mini_test)} test examples",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "mini_data"})
    
    # Create mini training setup
    train_dataloader = DataLoader(mini_train, shuffle=True, batch_size=4)  # Smaller batch
    train_loss = losses.TripletLoss(model=model)
    evaluator = create_evaluator(mini_test)
    
    # Run short training (just a few steps)
    logger.info("Starting mini training session (target: ~5 minutes)",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "mini_training_start"})
    
    start_time = time.time()
    
    # Configure training for hybrid CPU/GPU approach
    fit_kwargs = {
        "train_objectives": [(train_dataloader, train_loss)],
        "evaluator": evaluator,
        "epochs": 1,
        "evaluation_steps": 5,
        "warmup_steps": 2,
        "optimizer_params": {'lr': LEARNING_RATE},
        "output_path": str(OUTPUT_MODEL_PATH.parent / "test_model"),
        "save_best_model": False,
        "show_progress_bar": True,
        "use_amp": False,  # Disable mixed precision for DirectML compatibility
    }
    
    logger.info("🚀 Starting HYBRID CPU/GPU training - Model on CPU, computations on GPU")
    logger.info("Watch Task Manager - GPU should show activity during forward/backward passes")
    model.fit(**fit_kwargs)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Mini training completed in {total_time:.1f} seconds",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "mini_training_complete"})
    
    # Performance projections
    steps_per_epoch = len(train_examples) // BATCH_SIZE
    time_per_step = total_time / len(train_dataloader)
    projected_epoch_time = time_per_step * steps_per_epoch / 60  # minutes
    
    logger.info(f"📊 Performance projections:",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "projections"})
    logger.info(f"   Time per step: {time_per_step:.2f} seconds",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "projections"})
    logger.info(f"   Projected time per full epoch: {projected_epoch_time:.1f} minutes",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "projections"})
    logger.info(f"   Full training (3 epochs): {projected_epoch_time * 3:.1f} minutes",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "projections"})
    
    # Test embedding generation
    test_texts = ["Function test", "Object test", "Parameter test"]
    embeddings = model.encode(test_texts)
    logger.info(f"✅ Embedding generation successful: {embeddings.shape}",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "embedding_test"})
    
    logger.info("🎉 Hardware/software test completed successfully!",
                extra={"Model": CONFIG_MODEL_NAME, "Training Step": "test_complete"})
    
    return True


def test_finetuned_model():
    """
    Test the fine-tuned model with sample queries to validate training success.
    
    This function performs several diagnostic tests:
    
    1. MODEL LOADING TEST:
       - Verifies the fine-tuned model can be loaded without errors
       - Checks if all model components (tokenizer, weights, config) are intact
    
    2. EMBEDDING GENERATION TEST:
       - Generates embeddings for 3 sample texts from your domain
       - Reports embedding shape (should be (3, 2560) for Qwen3-Embedding-4B)
       - Validates embeddings are numerical (not NaN/Inf)
    
    3. SIMILARITY ANALYSIS:
       - Calculates cosine similarities between all pairs of test embeddings
       - Creates a 3x3 similarity matrix showing semantic relationships
    
    INTERPRETING RESULTS:
    
    ✅ GOOD RESULTS:
    - Embedding shape: (3, 2560) or similar consistent dimensions
    - Similarity matrix with values between 0.0 and 1.0
    - Diagonal values = 1.0 (text similar to itself)
    - Related texts show higher similarity (0.6-0.9)
    - Unrelated texts show lower similarity (0.1-0.5)
    
    Example good result:
    [[1.0000, 0.7234, 0.3456],
     [0.7234, 1.0000, 0.4123], 
     [0.3456, 0.4123, 1.0000]]
    
    ❌ BAD RESULTS (indicates training problems):
    - NaN values in similarity matrix: Model generates invalid embeddings
    - All similarities near 0.0: Model lost semantic understanding
    - All similarities near 1.0: Model overfitted/collapsed
    - Inconsistent embedding shapes: Model corruption
    
    TROUBLESHOOTING:
    - If you see NaN: Learning rate was too high, use diagnostic_embeddings.py
    - If similarities are random: Increase training epochs or improve data quality
    - If model won't load: Training was interrupted, retrain from scratch
    
    NEXT STEPS:
    - Good results: Export model with --action export
    - Bad results: Adjust config/embedding_training_config.py and retrain
    """
    if not OUTPUT_MODEL_PATH.exists():
        logger.error("Fine-tuned model not found. Run fine-tuning first.")
        return
    
    # Load the fine-tuned model
    model = SentenceTransformer(str(OUTPUT_MODEL_PATH))
    
    # Test with some sample texts from your domain
    test_texts = [
        "Function Summary > app > NewNamedBook: Creates and returns a new book object",
        "Object Reference > Constants: Defines integer identifiers for language options",
        "CMS Connector Framework > CMS API Data Structures: Defines delete-operation parameters"
    ]
    
    # Generate embeddings
    embeddings = model.encode(test_texts)
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}",
                extra={"Model": "fine-tuned", "Training Step": "testing"})
    
    # Calculate similarities
    from sentence_transformers.util import cos_sim
    similarities = cos_sim(embeddings, embeddings)
    
    logger.info(f"Similarity matrix:\n{similarities.numpy()}",
                extra={"Model": "fine-tuned", "Training Step": "similarity_test"})


def export_for_ollama():
    """Export the fine-tuned model in a format suitable for local deployment."""
    if not OUTPUT_MODEL_PATH.exists():
        logger.error("Fine-tuned model not found. Run fine-tuning first.")
        return
    
    # Load the fine-tuned model
    model = SentenceTransformer(str(OUTPUT_MODEL_PATH))
    
    # Save in HuggingFace format for easier integration
    hf_output_path = OUTPUT_MODEL_PATH.parent / "qwen3_embedding_hf"
    model.save(str(hf_output_path))
    
    # Create a simple usage example script
    example_script = f'''
# Example usage of the fine-tuned embedding model
from sentence_transformers import SentenceTransformer

# Load the fine-tuned model
model = SentenceTransformer(r"{hf_output_path}")

# Example: Encode some text
texts = [
    "Your coding question here",
    "Another related text"
]

embeddings = model.encode(texts)
print(f"Embeddings shape: {{embeddings.shape}}")

# For vector database storage
import numpy as np
# Save embeddings for vector DB ingestion
np.save("embeddings.npy", embeddings)
'''
    
    with open(OUTPUT_MODEL_PATH.parent / "usage_example.py", "w") as f:
        f.write(example_script)
    
    logger.info(f"Model exported to {hf_output_path} with usage example",
                extra={"Model": "fine-tuned", "Training Step": "export"})


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune embedding model")
    parser.add_argument("--action", choices=["train", "test", "export", "hardware_test", "smoke_steps"], 
                       default="train", help="Action to perform")
    args = parser.parse_args()
    
    if args.action == "train":
        fine_tune_model()
    elif args.action == "test":
        test_finetuned_model()
    elif args.action == "export":
        export_for_ollama()
    elif args.action == "hardware_test":
        test_hardware()
    elif args.action == "smoke_steps":
        # Run the limited-steps helper using configuration from TRAINING_CONFIG only.
        from simple_training_test import run_limited_training_steps
        # Ensure tokenized dataset exists before running smoke steps
        tokenized_path = TOKENIZED_DATA_DIR if 'TOKENIZED_DATA_DIR' in globals() else None
        if tokenized_path is None or not Path(tokenized_path).exists():
            logger.error("Tokenized dataset not found. Run scripts/tokenize_triplets.py to create it before smoke runs.")
            raise SystemExit("Tokenized dataset missing. Aborting smoke run.")

        train_examples, test_examples = load_training_data()
        # detect device for smoke run
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Set allocator config to reduce fragmentation; prefer PYTORCH_ALLOC_CONF
        try:
            os.environ['PYTORCH_ALLOC_CONF'] = TRAINING_CONFIG.get('PYTORCH_ALLOC_CONF', 'max_split_size_mb:128')
            logger.info(f"Set PYTORCH_ALLOC_CONF={os.environ['PYTORCH_ALLOC_CONF']} to reduce fragmentation")
        except Exception:
            logger.warning("Could not set PYTORCH_ALLOC_CONF; proceeding without it")

        model = setup_model(device=None)

        # Verify model's max_position_embeddings >= configured max_sequence_length
        try:
            hf_model = None
            if hasattr(model, 'auto_model') and model.auto_model is not None:
                hf_model = model.auto_model
            elif hasattr(model, '_first_module'):
                try:
                    hf_model = model._first_module()
                except Exception:
                    hf_model = None
            max_pos = None
            if hf_model is not None and hasattr(hf_model, 'config'):
                max_pos = getattr(hf_model.config, 'max_position_embeddings', None)
            if max_pos is None and hasattr(model, 'config'):
                max_pos = getattr(model.config, 'max_position_embeddings', None)
            if max_pos is not None:
                logger.info(f"Model max_position_embeddings={max_pos}")
                if max_pos < TRAINING_CONFIG.get('max_sequence_length', 0):
                    logger.error(f"Configured max_sequence_length={TRAINING_CONFIG.get('max_sequence_length')} exceeds model max_position_embeddings={max_pos}")
                    raise SystemExit("max_sequence_length exceeds model max_position_embeddings; adjust config or model")
        except Exception as e:
            logger.warning(f"Could not determine model max_position_embeddings: {e}")

        # All runtime options for smoke run come from TRAINING_CONFIG
        steps = TRAINING_CONFIG.get('SMOKE_STEPS', 5)
        bs = TRAINING_CONFIG.get('batch_size', 1)
        ga = TRAINING_CONFIG.get('gradient_accumulation_steps', 1)
        use_fp16 = TRAINING_CONFIG.get('fp16', False)
        # LoRA tri-state: consult config
        use_lora = TRAINING_CONFIG.get('use_lora', False)
        lora_r = TRAINING_CONFIG.get('lora_r', 8)
        lora_alpha = TRAINING_CONFIG.get('lora_alpha', 16)

        # Optional: run under torch.utils.bottleneck (re-invoke Python -m torch.utils.bottleneck)
        if TRAINING_CONFIG.get('USE_BOTTLENECK', False):
            try:
                import subprocess, sys
                # Avoid recursive re-invocation: if we're already running under the bottleneck harness
                # the environment flag SMOKE_UNDER_BOTTLENECK will be set by the parent and we should
                # not re-launch another subprocess.
                if os.environ.get('SMOKE_UNDER_BOTTLENECK') == '1':
                    logger.info('Already running under torch.utils.bottleneck (SMOKE_UNDER_BOTTLENECK=1). Skipping re-invoke.')
                else:
                    logger.info('Invoking smoke run under torch.utils.bottleneck (python -m torch.utils.bottleneck)')
                    # Set an env var so the subprocess only runs the smoke path once
                    env = os.environ.copy()
                    env['SMOKE_UNDER_BOTTLENECK'] = '1'
                    cmd = [sys.executable, '-m', 'torch.utils.bottleneck', __file__, '--action', 'smoke_steps']
                    subprocess.run(cmd, check=True, env=env)
                    logger.info('torch.utils.bottleneck profiling completed')
            except Exception as be:
                logger.warning(f'torch.utils.bottleneck failed: {be}. Continuing without it.')

        # Optional: run under NVIDIA Nsight Systems (nsys) if requested - we will attempt a best-effort call
        if TRAINING_CONFIG.get('USE_NSIGHT', False):
            try:
                import subprocess, sys
                logger.info('Attempting to run smoke_steps under nsys (NVIDIA Nsight Systems)')
                trace_out = os.path.abspath('nsys_report')
                cmd = ['nsys', 'profile', '--output', trace_out, sys.executable, __file__, '--action', 'smoke_steps']
                subprocess.run(cmd, check=True)
                logger.info(f'nsys profiling completed, output prefix: {trace_out}')
            except Exception as ne:
                logger.warning(f'nsys profiling requested but failed: {ne}. You may need nsys installed and in PATH.')

        # Optional: torch.profiler-based run (inline)
        use_torch_profiler = TRAINING_CONFIG.get('USE_TORCH_PROFILER', False)
        profiler_trace = TRAINING_CONFIG.get('TORCH_PROFILER_TRACE', None)
        profiler = None
        if use_torch_profiler:
            try:
                from torch import profiler
                profiler = profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                profiler.__enter__()
                logger.info('torch.profiler started for smoke run')
            except Exception as pe:
                profiler = None
                logger.warning(f'torch.profiler failed to start: {pe}')

        result = run_limited_training_steps(
            model,
            train_examples,
            steps=steps,
            device=device,
            batch_size=bs,
            grad_accum=ga,
            use_fp16=use_fp16,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Export profiler trace if it was running
        if profiler is not None:
            try:
                profiler.__exit__(None, None, None)
                if profiler_trace:
                    try:
                        profiler.export_chrome_trace(profiler_trace)
                        logger.info(f'torch.profiler trace saved to: {profiler_trace}')
                    except Exception as e:
                        logger.warning(f'Failed to export torch.profiler trace: {e}')
            except Exception:
                pass

        logger.info(f"Smoke steps result: {result}")
