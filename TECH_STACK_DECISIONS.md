# Tech Stack Decisions — Implemented Choices (current)

This document records the *current* runtime and tuning choices implemented in this repository and why they were selected. It intentionally excludes legacy/fallback implementations and focuses on the stack we actively use for GPU-accelerated fine-tuning and the limited-step smoke runner.

## Overview
- Primary compute: NVIDIA CUDA (PyTorch CUDA build). We target CUDA-only GPU acceleration on the local RTX 5060 Ti.
- Precision and memory optimizations: mixed precision (fp16 via AMP) and 8-bit optimizer states (bitsandbytes) where appropriate.
- Parameter-efficiency: LoRA (via the `peft` library) is provided as the recommended path for single-GPU fine-tuning of large embedding models.

All code paths now prefer CUDA when available and use the following combination to balance memory, speed, and embedding fidelity: LoRA adapters + fp16 (AMP) + bitsandbytes 8-bit optimizer (when training adapters or weights). If an OOM occurs, the runner falls back gracefully and produces diagnostics.

## Key dependencies (installed and used)
- torch (CUDA-enabled build) — primary runtime for model forward/backward and tensor ops.
- bitsandbytes — 8-bit optimizer support (e.g., `AdamW8bit`) to drastically reduce optimizer-state memory.
- peft — LoRA (PEFT) adapter application and utilities.
- sentence-transformers / transformers — model wrapper and underlying transformer backbone.

Installation example (done in the venv used by this project):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U peft bitsandbytes
```

## Decisions and rationale

### 1) Prefer CUDA (NVIDIA) for all heavy compute
- Why: CUDA gives native PyTorch acceleration and reliable performance on the RTX 5060 Ti.
- How we use it: model and tensors are moved to `torch.device('cuda')` when available. The smoke-runner and custom loops prefer CUDA and use autocast/GradScaler when fp16 is enabled.
- Impact: provides the largest single-step speedup for forward/backward, but requires careful memory management.

### 2) Mixed precision (fp16) via AMP + GradScaler
- Why: fp16 halves the memory required for activations and parameter copies (where supported) and often speeds up training on modern GPUs.
- How: we use PyTorch AMP (autocast) and `GradScaler` in the limited-run and training loops when `--fp16` is enabled.
- Impact on memory/perf: roughly 30–60% lower activation memory and faster tensor kernels on supported hardware. This allows larger batches or more activations per forward.
- Impact on embedding quality: with GradScaler and a conservative learning rate, fp16 typically preserves final quality close to fp32. However fp16 can be fragile (NaNs) if LR is too high or the model contains unstable ops. We added diagnostics and a one-time retry with fp16 disabled when NaNs are detected.

### 3) 8-bit optimizer (bitsandbytes)
- Why: optimizer state (Adam) requires two or more full-size tensors per parameter (exp_avg, exp_avg_sq). For large models this is the dominant memory cost. bitsandbytes stores those optimizer states in 8-bit, reducing memory by ~4x for optimizer state.
- How: when LoRA/adapters are used we prefer `bitsandbytes.optim.AdamW8bit` for adapter params if available; if not available we fall back to regular AdamW but with small parameter sets.
- Impact on memory/perf: large reduction in optimizer memory, enabling training that otherwise would OOM. It also tends to reduce host-to-device bandwidth for optimizer state. CPU cost is similar; bitsandbytes implements efficient kernels.
- Impact on quality: 8-bit optimizer state is an approximation but empirically often matches fp32 optimizer quality for reasonable LRs. Combine with fp16 weights for maximum memory reduction.

### 4) LoRA / PEFT adapters
- Why: full-weight fine-tuning of very large embedding models on a single 16GB GPU often fails due to optimizer memory. LoRA trains a small number of additional parameters (low-rank matrices) and keeps the base model frozen.
- How: we apply LoRA (via `peft`) to the underlying HF transformer submodule. The code injects adapters and ensures only adapter parameters are optimized.
- Impact on memory/perf: adapter-only training reduces gradient + optimizer memory proportionally to the adapter parameter count. This allows quick training iterations with tiny memory overhead and low optimizer state. Training time per step is lower than full-weight training because fewer parameters are updated.
- Impact on embedding quality: LoRA often reaches performance close to full-weight for many tasks, especially when domain shift is moderate, but may not always match full-weight quality for every dataset. It's a pragmatic tradeoff: much lower resource use for often-acceptable quality.

### 5) Prefetching: pre-tokenize + pinned-memory batches (new)

- What: We added a producer prefetcher that tokenizes upcoming micro-batches on CPU worker thread(s), converts tokenized dicts into pinned CPU tensors, and places them into a small FIFO queue. The training loop consumes these pre-pinned batches and moves them to GPU with non-blocking transfers.
- Why: This removes the tokenization/IO/GC work from the critical path between GPU steps. The GPU no longer waits for Python tokenization each iteration which smooths utilization and reduces idle spikes in Task Manager.
- Implemented functions / symbols:
  - `_to_pinned_tensor_dict(token_dict)` in `simple_training_test.py`: converts token dict entries into CPU pinned tensors for faster .to(device, non_blocking=True) transfers.
  - `_producer_thread_fn(batches)` in `simple_training_test.py`: background producer that tokenizes and enqueues pinned batches.
  - `prefetch_queue` (queue.Queue): small in-memory buffer of ready-to-run batches (default 4 batches).
  - `stop_producer` (threading.Event): used to gracefully stop the producer thread when the run completes.
  - Training loop changes inside `run_limited_training_steps`: it now fetches pre-tokenized batches from `prefetch_queue` and moves tensors to device using non_blocking transfers. It falls back to on-demand tokenization if the queue is empty.
- Expected benefits:
  - Much smoother GPU utilization with fewer idle spikes.
  - Lower per-step latency due to reduced CPU-side blocking between steps.
  - Small increase in host RAM usage (pinned memory) proportional to `PREFETCH_BATCHES * batch_size`.
- Trade-offs:
  - Slight code complexity and need to manage thread lifetime and pinned memory.
  - Pinned memory is a limited system resource — tune `PREFETCH_BATCHES` conservatively (default 4).

This change is conservative and opt-in within the limited-run harness; it does not affect other training paths unless they call `run_limited_training_steps`.

## 6) Additional tuning options implemented (1-5 requested)

The following options were implemented to further improve throughput and reduce CPU/GPU stalls. These are configurable via `TRAINING_CONFIG` in `config_embed_training.py`.

1) Use HF `AutoTokenizer` for batched tokenization
  - What: producer uses `transformers.AutoTokenizer` to tokenize lists of texts in bulk, which is faster and better optimized than repeated wrapper tokenization.
  - Where: `_producer_thread_fn` in `simple_training_test.py` will use `AutoTokenizer.from_pretrained(...)` when possible.

2) Prefetch workers
  - What: `TRAINING_CONFIG['PREFETCH_WORKERS']` controls how many producer workers to use. The default is 1 (thread). Using multiple processes avoids GIL constraints during tokenization.
  - Where: `simple_training_test.py` reads `PREFETCH_WORKERS` and starts the producer. For multiprocessing, future improvements can use `concurrent.futures.ProcessPoolExecutor`.

3) Torch.compile() optional wrapper
  - What: `TRAINING_CONFIG['USE_TORCH_COMPILE']` when True attempts to call `torch.compile(model)` to let PyTorch fuse kernels and reduce Python overhead.
  - Where: `embedding_finetuner.py::setup_model()` wraps the model when this flag is True.
  - Caveats: Requires PyTorch 2.x and can change memory/behavior; use only after verification.

4) Optional load_in_8bit + device_map
  - What: `TRAINING_CONFIG['USE_LOAD_IN_8BIT']` when True attempts a best-effort reload of the underlying HF model in 8-bit (`load_in_8bit=True, device_map='auto'`) and attaches it back to the SentenceTransformer wrapper.
  - Where: `embedding_finetuner.py::setup_model()` contains the loader logic.
  - Caveats: Requires `accelerate` and `bitsandbytes`; this is best-effort and falls back to normal load if unsupported.

5) Prefetch tuning and logging
  - New config keys: `PREFETCH_WORKERS` (int), `PREFETCH_BATCHES` (constant in harness, default 4). Adjust based on host RAM and CPU cores.

Files/functions to inspect for details:
- `simple_training_test.py`:
  - `run_limited_training_steps(...)` — now performs prefetching and consumes pinned batches.
  - `_to_pinned_tensor_dict(token_dict)` — converts token dict to pinned CPU tensors.
  - `_producer_thread_fn(batches)` — tokenizes using HF tokenizer when available and enqueues pinned batches.
  - `prefetch_queue` / `stop_producer` — queue and event for producer lifecycle.
- `embedding_finetuner.py`:
  - `setup_model()` — supports `USE_LOAD_IN_8BIT` and `USE_TORCH_COMPILE` flags.
- `config_embed_training.py`:
  - `TRAINING_CONFIG` now includes `USE_TORCH_COMPILE`, `USE_LOAD_IN_8BIT`, and `PREFETCH_WORKERS`.

Guidance for operators:
- Start conservative: keep `PREFETCH_BATCHES=4`, `PREFETCH_WORKERS=1` and monitor host memory. Increase `PREFETCH_WORKERS` to utilize more CPU cores for tokenization.
- If trying `USE_TORCH_COMPILE`, validate on a small run first and compare correctness and memory.
- If `USE_LOAD_IN_8BIT` is enabled, install `accelerate` and ensure your environment supports bitsandbytes; this path is optional and best-effort.

If you want, I can now:
- A) Replace the single-thread producer with a multiprocessing producer to bypass GIL and further speed tokenization.
- B) Reduce logging frequency to only sample DIAG every N steps to reduce CPU overhead.
- C) Implement a process-pool based producer and re-run a 20-step comparison.

Tell me which of A/B/C you prefer and I'll implement it and run a longer smoke test.

## Combined effect (LoRA + fp16 + bitsandbytes)
- Memory: 
  - fp16 reduces activation and parameter memory roughly 2x.
  - LoRA reduces gradients/optimizer state proportionally to the adapter size (often tiny: <0.1% of full model).
  - bitsandbytes reduces optimizer state (~4x less memory for Adam states).
  - Together these allow training on a 16GB GPU models that would otherwise need much more memory.
- Performance:
  - fp16 improves throughput on supported CUDA hardware.
  - Training adapters is faster per step due to fewer params to update.
  - bitsandbytes provides efficient optimizer updates with small memory footprint.
- Quality:
  - fp16 (with GradScaler) + bitsandbytes generally preserves final quality for adapter training.
  - LoRA may not perfectly match full-weight fine-tuning, but in practice yields strong results for embedding adaptation with much lower cost.

## Safety nets and diagnostics
- The limited-run includes:
  - Device-placement fixes (tokenized tensors moved to model device).
  - NaN/Inf diagnostics for embeddings (mean/std/min/max and any_nan checks).
  - One-time retry logic: if fp16 forward produces NaNs, the runner retries with fp16 disabled once.
  - OOM handling: if optimizer creation or step OOMs on CUDA, the runner can fallback to a CPU run to capture diagnostics rather than crash.

These measures reduce flakiness and provide clear logs to decide the next step.

## Practical recommendations / next steps
- Prefer LoRA + fp16 + bitsandbytes for single-GPU fine-tuning of large embedding models.
  - Start with LoRA rank r=8 and alpha=16 and monitor validation metrics.
  - If memory is still tight, reduce r to 4 or increase gradient accumulation.
- If you need absolutely maximum accuracy and have cluster/multi-GPU resources, use ZeRO/offload (DeepSpeed/Accelerate) to run full-weight fp32/fp16 training.
- Monitor for NaNs on early steps. If you see NaNs:
  - Lower learning rate and retry.
  - Disable fp16 and test forward/backward in fp32 to isolate the problem.
  - Verify the checkpoint integrity (run `diagnose_model_forward.py`).

## Files of interest
- `simple_training_test.py` — limited-run smoke function and LoRA integration.
- `embedding_finetuner.py` — CLI wiring (flags: `--fp16`, `--use_lora`, `--lora_r`, `--lora_alpha`).
- `diagnose_model_forward.py` — quick diagnostic helper used to validate model forward outputs.

## Tokenization pipeline (new)

We added a one-time tokenization pipeline to remove repeated host-side tokenization stalls and make repeated smoke/benchmark runs fast and deterministic.

- Script: `scripts/tokenize_triplets.py`
  - Purpose: tokenize the training triplets CSV once and save a tokenized dataset using `datasets.save_to_disk()`.
  - Config constants at the top of the script (edit as needed):
    - `MODEL_NAME` — which HF/SentenceTransformers tokenizer to use
    - `TRAIN_CSV` — input CSV filename (expects `anchor`, `positive`, `negative` columns)
    - `MAX_LENGTH` — token truncation/padding length
    - `PADDING` — padding strategy (default `max_length` in the script)
    - `NUM_PROC` — parallel workers for `datasets.map()` (set smaller on Windows)
    - `BATCH_SIZE` — tokenization batch size
  - Outputs:
    - tokenized dataset directory: `tokenized_train/` (saved via `save_to_disk`)
    - `metadata.json` in the tokenized dir with tokenizer name, max_length, padding, num_proc, batch_size, `avg_anchor_length`, `p90_anchor_length`, `p95_anchor_length`, and `num_examples`.

- Why: tokenizing once eliminates the CPU tokenization cost from the training critical path. The saved dataset can be loaded quickly and consumed by the training loop with minimal overhead.

- Best practices:
  - Use `num_proc` > 1 on Linux for parallel tokenization; on Windows prefer `num_proc=1` or run the script as a script (not interactive) to avoid multiprocessing issues.
  - Choose `MAX_LENGTH` using the saved `metadata.json` (avg/p90/p95) — a common approach is to set `max_length` to the 95th percentile rounded to a multiple of 8.
  - Prefer saving tokenized arrays (no torch tensors) and performing dynamic padding in the DataLoader via a DataCollator if you want smaller disk size.

- Finetuner check: `embedding_finetuner.py` now validates that `TOKENIZED_DATA_DIR` exists before running `fine_tune_model()` or `--action smoke_steps`. If missing, it aborts with a clear message and a suggested command:
  ```powershell
  python scripts/tokenize_triplets.py
  ```

- How to run the tokenizer (example):
  ```powershell
  .venv\Scripts\Activate.ps1
  python scripts/tokenize_triplets.py
  ```

Adding this pipeline significantly reduces per-step variance and host-side stalls. After tokenizing, re-run your 5-step smoke test to confirm per-step time and variance improvements.

## Prepadding rationale

Short answer: pad to 88 is a good choice — it’s simple, aligned to hardware (88 is divisible by 8), and will capture ~95%+ of your examples while cutting wasted work vs 256 by a large margin.

Why pre-padding to 88 is sensible here

Your stats (avg ≈56, p90=80, p95=84) show almost all examples are ≤88, so truncation loss will be tiny.
Pre-padding to a fixed, small length gives predictable peak memory and removes per-batch padding work at training time. That helps the prefetch/pinned-tensor pipeline you already implemented: you can create CPU-pinned tensors once (already padded), queue them, and move them to GPU non-blocking. That minimizes CPU overhead and GPU stalls.
88 is a multiple of 8 (good for some GPU kernels) so performance is fine; if you wanted to be ultra-conservative 96 is also fine — but 88 is compact and safe given your p95.
Tradeoffs vs dynamic padding (DataCollator / batch-longest)

Compute: Dynamic padding (pad-to-batch-longest) will reduce the total tokens processed slightly more than fixed-88 when batches have many very-short examples, because it never does padding to the fixed maximum. So it can reduce FLOPS and slightly reduce runtime.
Memory (peak): Dynamic padding can reduce per-batch memory if batches are skewed short. But peak memory is driven by the largest batch member — if you still include some long examples in a batch, peak won't drop that much. Fixed-88 gives deterministic peak which helps avoid OOM surprises.
Overhead/complexity: Dynamic padding requires a collator on every batch (CPU work) and potentially varies tensor shapes each batch. That adds small per-batch CPU cost and slightly more complexity to keep tensors pinned and prefetch working efficiently. If you already moved tokenization off the hot path and have a pinned-buffer prefetch pipeline, pre-padding is simpler and likely faster in practice.
Implementation safety: Pre-padding lets you create the dataset once, reuse it, and keeps the training loop simpler and more deterministic.
Recommendation

Given your observed lengths, pre-padding to 88 is the pragmatic, low-complexity win: it reduces compute and memory dramatically vs 256, keeps the prefetch pipeline simple, and only truncates a tiny tail of examples.
If later you want to squeeze a bit more compute out, you can switch to a DataCollatorWithPadding to use dynamic per-batch padding; but test to confirm the extra CPU collate overhead doesn’t offset the token savings on your hardware.



# Embedding model tuning: https://www.sbert.net/docs/usage/finetuning.html
# https://medium.com/@diagnosta/lora-fine-tuning-of-embedding-models-using-llamaindex-a60b823a2c94 Mar 18, 2024
# https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/
# https://medium.com/@tuhinsharma121/fine-tuning-embedding-models-with-llamaindex-a-hands-on-guide-with-hugging-face-and-proprietary-ae1732dc814a

# Process
# Install a local model (bi-encoder). For embedding model rankings, see https://huggingface.co/spaces/mteb/leaderboard
# Prepare training data: triplets of (anchor, positive, negative) sentences. Anchor and positive are semantically similar; negative is not.
# Fine-tune the model using the triplet loss function.
# Evaluate the model using a relevant dataset.
# Use the fine-tuned model to generate embeddings for your documents at Int8 or desired precision.
# Note: You can train the embedding model and the reranker model separately or together using SBERT. 





# Current todo list 

Done: summarize extend script and mif
for 10/21: generate training data from extendscript and mif: create, tokenize, test
create easy or semi-hard negatives (random or BM25-based)
run first epoch
create embeddings from generic LLM/agent
run two more epochs
train my embedding model
delete the original embeddings
create new ones from my trained embedding model

create reranker training data
train my reranker model
upload into vector db. 

=====================

combined_data = dataset1 + dataset2 # Combine datasets and interleave
random.shuffle(combined_data)    	# Reshuffle after each epoch
train_model(combined_data, epochs=3)

loader = MultiDatasetDataLoader(
    datasets=[loaderA, loaderB],
    sampling_ratios=[0.5, 0.5] # sampling weights (e.g., 50/50; not raw size).
)

# Mine/insert hard negatives across sets if possible; it improves a shared space.
# Build embeddings with current model, index, then mine per epoch
emb = model.encode(corpus_union, convert_to_tensor=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.cpu().numpy())
for (a,p,_) in triplets:  # ignore old negative
    D,I = index.search(a_emb, k=50)
    neg = first_non_match(I, same_group=False)  # prefer other set
    new_triplets.append((a, p, neg))
# Train TripletLoss on new_triplets; repeat next epoch.



Stop/measure: early-stop on per-domain R@k and MRR to avoid overfitting one set.

During training, evaluate separately on each dataset’s dev set to make sure your model improves both—and doesn’t overfit one.

- R@k (Recall@k): checks if the correct match is among the top-k retrieved items.
- MRR (Mean Reciprocal Rank): measures average 1/rank of the correct match.

Example: every epoch, run retrieval eval on devA and devB; if devA keeps rising but devB drops, stop early or rebalance sampling. This prevents one dataset from dominating the shared embedding space.



============= triplet eval ===========

✅ Your Negatives Are Good Quality - You Can Start Training
Why They're Good:
Example 1:

Anchor: "Array of TypedVal objects..."
Positive: "Removes last element... Returns: TypedVal"
Negative: "Removes last element... Returns: Tab"
Quality: Semi-hard - Same function (pop), but different return type. This is excellent! The model must learn the difference between TypedVal and Tab arrays.
Example 2:

Anchor: About Pantone color constants
Positive: About constants and formatting flags
Negative: Just a list: "Delete, GetProps, MoveComponent..."
Quality: Easy-medium - Both are about FrameMaker concepts, but negative lacks semantic connection to color/constants. Good enough.
Example 3:

Anchor: Array of objects
Positive: concat function for Ints
Negative: GetExportDefaultParams function
Quality: Medium - Both are functions, but semantically unrelated (arrays vs export). Good challenge.
Example 4:

Anchor: Notifications system overview
Positive: Notifications handler example
Negative: GetProps for Group
Quality: Medium-hard - Negative is same domain but completely different topic. Good.
Assessment:
✅ Your negatives are well-constructed - they're semi-hard (same domain, but semantically different)

✅ Mix of difficulty levels - some are subtle (Example 1), others more obvious

✅ All from same domain - this is actually good for your use case (FrameMaker/JSX documentation)

Recommendation:
Skip the negative generation step - start training now!

Your negatives are good enough for first-pass training. You can always implement hard negative mining between epochs (as your document suggests) if you want to improve quality later.

=========================================
training gigs rquired: Param X 

Model size (GB) ≈ Parameters × bytes per parameter ÷ 1 billion

Param times quant value (32 =1) 

=========

## Creating training data

Why generate all difficulties at once:

Single pass through data - You only need to load and process your chunks once
Consistent strategy - All negatives use the same criteria/logic
Time efficient - Don't need to re-run the script multiple times
Data consistency - All epochs use the same positive/anchor pairs, just different negatives
Easy comparison - You can see the difficulty progression clearly
How to structure it:


==============

ok. So now I need to train a reranker. Can i use the same training data?

Yes! You can use the same triplets data to train a reranker. In fact, it's the ideal approach:

Same data structure works:

Reranker also learns from anchor/positive/negative triplets
Your existing triplets_train.json files are perfect
The difference is in what the model learns:
Embeddings: Maps text → vectors (semantic similarity in vector space)
Reranker: Takes (query, document) pairs → relevance score (0-1)
Reranker training differences:

Uses a cross-encoder architecture (not bi-encoder like embeddings)
Sees both query and document together (more context)
Slower at inference but more accurate
Typically smaller models work well (BERT-base, etc.)
Your workflow:

✅ Trained embedding model (done)
🔜 Train reranker on same triplets
🔜 Deploy: Embeddings retrieve top-K → Reranker picks best
Next step:
Would you like me to help you set up a reranker training script? It would be similar to your embedding trainer but using cross-encoder architecture.


===============

================================================================================
📈 RESULTS
================================================================================
Accuracy:               82.13% (positive closer than negative)
Mean Positive Similarity: 0.8618 (higher is better)
Mean Negative Similarity: 0.3932 (lower is better)
Margin (pos - neg):      0.4686 (higher is better)




✅ Training epoch 2 - Training with medium negatives (harder examples) completed successfully
INFO: Using pre-tokenized DataLoader fast path
INFO: Batch 0, Loss: 0.4904, GPU: Active
INFO: Batch 10, Loss: 0.7701, GPU: Active
INFO: Batch 20, Loss: 0.7412, GPU: Active
INFO: Batch 30, Loss: 0.7276, GPU: Active
INFO: Batch 40, Loss: 0.7150, GPU: Active
INFO: Batch 50, Loss: 0.7032, GPU: Active
INFO: Batch 60, Loss: 0.6913, GPU: Active
INFO: Batch 70, Loss: 0.6822, GPU: Active
INFO: Batch 80, Loss: 0.6646, GPU: Active
INFO: Batch 90, Loss: 0.6436, GPU: Active
INFO: Batch 100, Loss: 0.6175, GPU: Active
INFO: Batch 110, Loss: 0.5963, GPU: Active
INFO: Batch 120, Loss: 0.5697, GPU: Active
INFO: Batch 130, Loss: 0.5490, GPU: Active
INFO: Batch 140, Loss: 0.5283, GPU: Active
INFO: Batch 150, Loss: 0.5072, GPU: Active
INFO: Batch 160, Loss: 0.4895, GPU: Active
INFO: Batch 170, Loss: 0.4712, GPU: Active
INFO: Batch 180, Loss: 0.4549, GPU: Active
INFO: Batch 190, Loss: 0.4422, GPU: Active
INFO: Batch 200, Loss: 0.4323, GPU: Active
INFO: Batch 210, Loss: 0.4229, GPU: Active
INFO: Batch 220, Loss: 0.4143, GPU: Active
INFO: Batch 230, Loss: 0.4073, GPU: Active

✅ Training epoch 3 - Training with hard negatives (challenging examples) completed successfully
INFO: Using pre-tokenized DataLoader fast path
INFO: Batch 0, Loss: 0.8251, GPU: Active
INFO: Batch 10, Loss: 0.8044, GPU: Active
INFO: Batch 20, Loss: 0.8437, GPU: Active
INFO: Batch 30, Loss: 0.8698, GPU: Active
INFO: Batch 40, Loss: 0.8832, GPU: Active
INFO: Batch 50, Loss: 0.8839, GPU: Active
INFO: Batch 60, Loss: 0.8710, GPU: Active
INFO: Batch 70, Loss: 0.8701, GPU: Active
INFO: Batch 80, Loss: 0.8623, GPU: Active
INFO: Batch 90, Loss: 0.8498, GPU: Active
INFO: Batch 100, Loss: 0.8348, GPU: Active
INFO: Batch 110, Loss: 0.8182, GPU: Active
INFO: Batch 120, Loss: 0.7958, GPU: Active
INFO: Batch 130, Loss: 0.7707, GPU: Active
INFO: Batch 140, Loss: 0.7615, GPU: Active
INFO: Batch 150, Loss: 0.7455, GPU: Active
INFO: Batch 160, Loss: 0.7349, GPU: Active
INFO: Batch 170, Loss: 0.7158, GPU: Active
INFO: Batch 180, Loss: 0.6961, GPU: Active
INFO: Batch 190, Loss: 0.6934, GPU: Active
INFO: Batch 200, Loss: 0.6798, GPU: Active
INFO: Batch 210, Loss: 0.6773, GPU: Active
INFO: Batch 220, Loss: 0.6693, GPU: Active
INFO: Batch 230, Loss: 0.6574, GPU: Active

