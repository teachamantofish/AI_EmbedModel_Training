# Configuration for embedding model fine-tuning
# Get a model: To fine-tune and then produce embeddings locally get
# a HuggingFace/SentenceTransformers-style model folder (a directory 
# with model weights, config, tokenizer, etc.) or a Transformers model.
# Base configuration
from pathlib import Path
import os

# Multi-epoch training configuration
# Set CURRENT_EPOCH manually when running or resuming training
CURRENT_EPOCH = 1  # Which epoch to run (1=easy, 2=medium, 3=hard)
RUN_EPOCHS = [1, 2, 3]  # List of all epochs to run in sequence

BASE_MODEL = r"C:\Users\benro\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-4B\snapshots\5cf2132abc99cad020ac570b19d031efec650f2b"
BASE_CWD = Path("C:/GIT/AI_DataSource/framemaker") # Base domain working directory. May contain 1-N docs.
OUTPUT_MODEL_PATH = BASE_CWD / "qwen3-embedding-4b-finetuned" # Output path for fine-tuned model
DOCLIST = ["extendscript", "mifref"] # Training data source dirs. The orchestrator can iterate through these to combine datasets
DOCDIR = DOCLIST[0] # DOCDIR: Current active directory (defaults to first in DOCLIST)
LOG_FILES = BASE_CWD / DOCDIR
TRAINING_DATA_DIR = BASE_CWD / DOCDIR / "embedding_training_data" # Training data (base path - difficulty subdirectory added by pipeline
CONFIG_MODEL_NAME = BASE_MODEL # Model to use for tokenization and training (will be set by pipeline based on CURRENT_EPOCH)

# Reranker configuration
RERANKER_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Small, fast reranker
RERANKER_OUTPUT_PATH = BASE_CWD / "framemaker-reranker-finetuned"

# Training parameters
# PERFORMANCE IMPACT GUIDE:
# ========================
# Model choice: PyTorch training requires the model in HuggingFace format, so 
# CONFIG_MODEL_NAME must point to a valid model directory.
# These settings directly affect training time, memory usage, and model quality.
# Estimates based on Qwen3-4B with 730 triplets on 60GB RAM system:
#
# EPOCHS: Number of complete passes through training data
# - 1 epoch: ~3-4 hours (CPU), basic learning
# - 3 epochs: ~10-12 hours (CPU), good learning  
# - 5+ epochs: ~15+ hours (CPU), risk of overfitting
# - Memory impact: None (same data, multiple passes)
# - Per 1000 triplets: Add ~40% to above times
#
# CONTINUING TRAINING (Multi-Epoch Strategy):
# To continue training from a previous epoch:
# 1. After epoch 1 completes, the model is saved to OUTPUT_MODEL_PATH (configured in config)
# 2. Update CONFIG_MODEL_NAME in config to point to the saved model from epoch 1
# 3. Run training again for epoch 2 - it will load from your fine-tuned model
# 4. Repeat: Update CONFIG_MODEL_NAME to epoch 2 output, run epoch 3
# 5. This allows progressive refinement with harder negatives or additional data
#
# Example:
#   Epoch 1: CONFIG_MODEL_NAME = "C:\\.cache\\huggingface\\...\\Qwen3-Embedding-4B"
#   Epoch 2: CONFIG_MODEL_NAME = "C:\\...\\qwen3-embedding-4b-finetuned"  # from epoch 1
#   Epoch 3: CONFIG_MODEL_NAME = "C:\\...\\qwen3-embedding-4b-finetuned-epoch2"  # from epoch 2
#
# LEARNING_RATE: How aggressively model weights are updated
# - 1e-6 (very low): Stable, slow learning, may underfit
# - 5e-6 (current): Conservative, prevents NaN, longer training
# - 2e-5 (normal): Faster learning, risk of instability/NaN
# - 1e-4+ (high): Fast but likely to cause NaN or poor quality
# - Training time impact: Minimal (same computation)
# - Quality impact: Too low = underfit, too high = NaN/overfit
#
# BATCH_SIZE: How many triplets processed simultaneously
# - Memory formula: ~2GB × batch_size for Qwen3-4B
# - Per 1000 triplets: Time scales roughly inversely with batch size
# - Effective batch size = batch_size * gradient_accumulation_steps
#
# FP16: Half-precision training (16-bit vs 32-bit floats)
# - False (current): Stable, full memory usage, prevents NaN
# - True: 2x faster, 50% less memory, but may cause NaN with large models
# - Memory savings: ~30-50% reduction
# - Time savings: ~50% faster training
# - Risk: Higher chance of NaN embeddings
#
# GRADIENT_ACCUMULATION_STEPS: Simulate larger batches without memory cost
# - 1: No accumulation, memory efficient
# - 4 (current): Simulates 4x batch size (8×4=32 effective batch)
# - 8+: Very large effective batch, may slow convergence
# - Memory impact: Minimal increase
# - Training impact: More stable gradients, slightly slower per step
#
# max_sequence_length impact (practical):
# - Attention memory can grow O(seq_len^2) for standard full attention (biggest memory pressure).
# - Token buffers and activations scale ~O(seq_len); doubling seq_len roughly doubles per-step compute.
# - Longer seqs increase tokenization memory and slow each training step; pick the shortest seq that preserves signal.
# - Mitigations: truncate/chunk long docs, use sliding-window or memory-efficient attention, 
# enable gradient_checkpointing/fp16 when safe. 
#
# MEMORY REQUIREMENTS (Qwen3-4B):
# - Base model: ~15GB
# - Training overhead: ~batch_size × 2GB
# - Total needed: ~15GB + (batch_size × 2GB)

TRAINING_CONFIG = {
    "epochs": 1,  # Number of passes through the training data.
    "learning_rate": 2e-5,  # Higher LR for faster convergence on smaller model
    "warmup_steps": 50,  # Shorter warmup
    "evaluation_steps": 100,  # Number of steps between evaluations: no vram impact.
    "save_steps": 500, # Number of steps between model saves: no vram impact.
    "fp16": True,  # Enable fp16 to reduce memory usage by ~50%
    "gradient_accumulation_steps": 4,  # Interacts with batch size. Allows simulating larger batches with no memory buildup
    "batch_size": 5,  # Raises mem usage, but larger batches imporve training quality by averaging gradients over more examples.
    "max_sequence_length": 256,  # Maximum token length for inputs; adjust based on your data
    # NOTE: This should never exceed the model's `config.max_position_embeddings`.
    #       Verify the local model's config (model.config.max_position_embeddings) is >= this value.
    #       E.g., for Qwen-3 variants this can be very large (tens of thousands) but always check.
    # Fail-fast and timeout controls
    # If True, any attempt to fall back to CPU (for optimizer state/allocation or retry) will abort the process.
    "FAIL_ON_CPU_FALLBACK": False,
    # Max allowed time (seconds) for a single training step. If exceeded, the script will abort.
    "MAX_STEP_SECONDS": 120,
    # Max total time (seconds) for a limited run. If exceeded, the script will abort.
    "MAX_TOTAL_SECONDS": 900,
    # Optional advanced performance flags
    # If True, attempt to use torch.compile() on the model (PyTorch 2.x). Test first. May causes issues with other code.
    "USE_TORCH_COMPILE": False,
    # If True, attempt to load the underlying HF model in 8-bit with device_map='auto'. Requires bitsandbytes & accelerate.
    # Disabled by default to prefer FP32 behavior and avoid allocator/fragmentation changes.
    "USE_LOAD_IN_8BIT": True,
    # Number of prefetch worker processes/threads used by the limited-run prefetcher. Default 1 (thread), set higher to use more CPU.
    "PREFETCH_WORKERS": 0,
    "DATALOADER_PIN_MEMORY": True,
    # How many steps to run for smoke tests (limited-step runs)
    "SMOKE_STEPS": 6,
    # Diagnostics/profiling flags
    "USE_BOTTLENECK": False,  # If True, run the smoke test under `python -m torch.utils.bottleneck`
    # Disable the heavy profiler for normal FP32 smoke runs to avoid runtime overhead and CUPTI issues on Windows
    "USE_TORCH_PROFILER": False,
    "TORCH_PROFILER_TRACE": None,
    "USE_NSIGHT": False,  # If True, attempt to invoke NVIDIA Nsight (nsys) to profile the smoke run (external tool)
    # How often (every N steps) to emit detailed DIAG logs (isfinite/mean/std).
    # Set to >1 (e.g., 5) to reduce CPU logging overhead during long runs.
    "DIAG_EVERY_N_STEPS": 5,
    # LoRA (PEFT) defaults - opt-in via config or CLI flag
    "use_lora": True,
    "lora_r": 12, # (rank) — # of trainable dimensions added to each weight matrix. < 16 = faster, less VRAM > = more VRAM better quality. 
    "lora_alpha": 24, # scaling factor: controls how strongly LoRA modifies the base weights. Typical default = alpha = 2 * r
    "lora_dropout": 0.1,  # Dropout probability for LoRA layers
    "lora_target_modules": ["q_proj", "v_proj"],  # Which layers to apply LoRA to (attention query/value projections)
    # Triplet loss configuration
    "triplet_margin": 1.0,  # Margin for triplet loss (distance between positive and negative)
    # Model loading configuration
    "trust_remote_code": True,  # Whether to trust remote code when loading models (required for some models)
}

# Tokenized dataset location (created by scripts/tokenize_triplets.py)
TOKENIZED_DATA_DIR = TRAINING_DATA_DIR / "tokenized_HFTrainer"

# Ollama integration settings
OLLAMA_EXPORT = {
    "quantization": "Q8_0",  # 8-bit quantization (matches your int8 preference)
    "embedding_dimension": 1024,  # Truncate to your preferred dimension
    "context_length": 2048,  # Context window for the model
    "temperature": 0.1,  # Low temperature for consistent embeddings
    "export_gguf": True,  # Export in GGUF format for Ollama
    "normalize_before_quantization": True,  # Better int8 performance
}

# Vector database integration
VECTOR_DB_CONFIG = {
    "embedding_dimension": 1024,  # Your preferred dimension for Qwen3-Embedding-4B
    "quantization": "int8",  # Quantize to int8 for storage efficiency
    "similarity_metric": "cosine",
    "index_type": "hnsw",  # Hierarchical Navigable Small World
    "ef_construction": 200,  # HNSW parameter
    "m": 16,  # HNSW parameter
    "normalize_embeddings": True,  # Normalize before quantization for better int8 performance
}


RERANKER_TRAINING_CONFIG = {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 16,  # Cross-encoders use more memory per batch
    "warmup_steps": 100,
    "max_length": 512,  # Query + document together
    "use_amp": True,  # Automatic mixed precision for speed
}