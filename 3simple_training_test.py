#!/usr/bin/env python3
"""
SIMPLE TRAINING TEST - Bridge between working GPU tests and full training
========================================================================

Test just the core training components that work in our GPU tests.

Run simple_training_test.py first whenever you want a quick check that the Python environment, 
CUDA/DirectML, and the sentence-transformers runtime are OK.

Run ONCE
"""

import torch
import os
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from scripts.config_embed_training import *
from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)

def simple_training_test():
    """Test basic training components with GPU acceleration."""
    logger.info("🎯 SIMPLE TRAINING TEST - Core Components Only")
    
    try:
        # 1. Device selection: prefer CUDA (NVIDIA), then DirectML, then CPU
        device = None
        device_name = "cpu"
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f'cuda:{torch.cuda.current_device()}'
            logger.info(f"✅ CUDA available. Using device: {device_name}")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                device_name = str(device)
                logger.info(f"✅ DirectML available. Using device: {device_name}")
            except Exception:
                device = torch.device('cpu')
                device_name = 'cpu'
                logger.info("ℹ️ No GPU acceleration available. Using CPU")
        
        # 2. Load small model (sentence-transformers/all-MiniLM-L6-v2 is tiny)
        logger.info("📦 Loading small test model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✅ Model loaded on CPU")

        # 3. Test basic encoding (no gradients yet)
        logger.info("🧪 Testing basic encoding...")
        test_texts = ["Hello world", "Good morning"]
        embeddings = model.encode(test_texts, convert_to_tensor=True)
        logger.info(f"✅ Embeddings shape: {embeddings.shape}")

        # 4. Move embeddings to selected device (GPU or CPU)
        logger.info(f"🚀 Moving embeddings to device: {device_name}...")
        gpu_embeddings = embeddings.to(device)
        logger.info(f"✅ Device embeddings shape: {gpu_embeddings.shape}")

        # 5. Test basic loss computation on GPU
        logger.info("📊 Testing loss computation on GPU...")
        anchor = gpu_embeddings[0:1]
        positive = gpu_embeddings[1:2] 
        negative = gpu_embeddings[0:1]  # Fake negative for test

        loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
        logger.info(f"✅ Loss computed on GPU: {loss.item()}")

        logger.info("🎉 SIMPLE TEST PASSED - All core components work!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple test failed: {e}")
        return False

if __name__ == "__main__":
    simple_training_test()

def run_limited_training_steps(model, train_examples, steps: int = 5, device=None, batch_size: int = 1, grad_accum: int = 1, use_fp16: bool = False, tried_fallback: bool = False, use_lora: bool = None, lora_r: int = 8, lora_alpha: int = 16):
    """Run a short training loop for `steps` optimizer updates (not batches).

    Args:
        model: SentenceTransformer model
        train_examples: list of InputExample-like objects (texts list)
        steps: number of optimizer steps to perform
        device: torch.device for computations
        batch_size: per-step micro batch size
        grad_accum: gradient accumulation steps (effective batch = batch_size * grad_accum)
    """
    import time
    from torch.optim import AdamW
    from sentence_transformers import losses
    from torch.utils.data import DataLoader
    import torch
    import threading
    import queue

    logger.info(f"Starting limited run: steps={steps}, batch_size={batch_size}, grad_accum={grad_accum}")

    # --- MODEL POSITION EMBEDDINGS CHECK -------------------------------------------------
    try:
        # Attempt to find underlying HF model config (works if SentenceTransformer wraps HF model)
        hf_model = None
        try:
            if hasattr(model, 'auto_model') and model.auto_model is not None:
                hf_model = model.auto_model
        except Exception:
            hf_model = None
        if hf_model is None and hasattr(model, '_first_module'):
            try:
                hf_model = model._first_module()
            except Exception:
                hf_model = None
        max_pos = None
        if hf_model is not None and hasattr(hf_model, 'config'):
            max_pos = getattr(hf_model.config, 'max_position_embeddings', None)
        # Fallback: check model.config if present
        if max_pos is None and hasattr(model, 'config'):
            max_pos = getattr(model.config, 'max_position_embeddings', None)
        if max_pos is not None:
            logger.info(f"Model max_position_embeddings={max_pos}")
            if max_pos < TRAINING_CONFIG.get('max_sequence_length', 0):
                logger.error(f"Configured max_sequence_length={TRAINING_CONFIG.get('max_sequence_length')} exceeds model max_position_embeddings={max_pos}")
                raise RuntimeError("max_sequence_length exceeds model max_position_embeddings; adjust config or model")
    except Exception as e:
        # Do not block training on inability to introspect; just warn
        logger.warning(f"Could not verify model max_position_embeddings: {e}")
    # ------------------------------------------------------------------------------------

    # Resolve LoRA defaults: if use_lora is None, consult config; otherwise use explicit bool
    if use_lora is None:
        effective_use_lora = TRAINING_CONFIG.get('use_lora', False)
    else:
        effective_use_lora = bool(use_lora)

    effective_lora_r = lora_r if lora_r is not None else TRAINING_CONFIG.get('lora_r', 8)
    effective_lora_alpha = lora_alpha if lora_alpha is not None else TRAINING_CONFIG.get('lora_alpha', 16)

    # Print effective runtime options for clarity
    logger.info(f"Runtime options: use_fp16={use_fp16}, requested_device={device}, use_lora={effective_use_lora}, lora_r={effective_lora_r}, lora_alpha={effective_lora_alpha}")

    # Ensure we have a device object
    if device is None:
        device = torch.device('cpu')

    # Enable some GPU performance flags when running on CUDA
    def _enable_gpu_tuning(device):
        try:
            if device is not None and getattr(device, 'type', None) == 'cuda':
                # Allow cuDNN autotuner to pick best algorithms for fixed-size inputs
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                # Enable TF32 where supported (tensor core speedups on modern GPUs)
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                except Exception:
                    pass
                try:
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
                logger.info("🔧 GPU tuning enabled: cudnn.benchmark=True, allow_tf32=True (if available)")
        except Exception:
            logger.debug("GPU tuning flags could not be applied")

    _enable_gpu_tuning(device)

    # Move model to the target device so forward/backward happen on the same device
    try:
        # Optional: run under torch.utils.bottleneck if configured. This will re-run the entire
        # limited run under the bottleneck harness by invoking the module as a subprocess.
        if TRAINING_CONFIG.get('USE_BOTTLENECK', False):
            try:
                import subprocess, sys
                # Prevent recursion: if we're already running under the bottleneck harness
                # the env var RUN_LIMITED_UNDER_BOTTLENECK will be set and we should not re-invoke.
                if os.environ.get('RUN_LIMITED_UNDER_BOTTLENECK') == '1':
                    logger.info('Already running under torch.utils.bottleneck (RUN_LIMITED_UNDER_BOTTLENECK=1). Skipping re-invoke.')
                else:
                    logger.info('Running limited run under torch.utils.bottleneck (this will re-invoke Python)')
                    # Build the python -m torch.utils.bottleneck invocation for this script
                    cmd = [sys.executable, '-m', 'torch.utils.bottleneck', __file__]
                    # We pass an environment flag so the subprocess runs only the limited run
                    env = os.environ.copy()
                    env['RUN_LIMITED_UNDER_BOTTLENECK'] = '1'
                    subprocess.run(cmd, check=True, env=env)
                    logger.info('torch.utils.bottleneck run complete')
                    return {'bottleneck': True}
            except Exception as be:
                logger.warning(f'torch.utils.bottleneck run failed: {be}. Continuing with normal limited run.')

        # Optional: wrap with torch.profiler if requested in config
        use_torch_profiler = TRAINING_CONFIG.get('USE_TORCH_PROFILER', False)
        profiler = None
        if use_torch_profiler:
            try:
                from torch import profiler
                # Simple profiler config: record operator and CUDA activity
                profiler = profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False
                )
                profiler.__enter__()
                logger.info('torch.profiler started')
            except Exception as pe:
                logger.warning(f'torch.profiler not available or failed to start: {pe}')
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
    except Exception:
        # Some SentenceTransformer wrappers may not implement .to(); ignore if so
        logger.debug("Model .to() failed or not available; proceeding (model may already be on device)")

    # Optionally prepare LoRA adapters + 8-bit optimizer support
    lora_applied = False
    # Detect bitsandbytes/AdamW8bit once and expose as local variable
    AdamW8bit = None
    try:
        import bitsandbytes as bnb  # noqa: F401
        from bitsandbytes.optim import AdamW8bit as _AdamW8bit
        AdamW8bit = _AdamW8bit
        logger.info("bnb: bitsandbytes detected; AdamW8bit available for optimizer")
    except Exception:
        logger.info("bnb: bitsandbytes/AdamW8bit not available; will use standard AdamW")
    if effective_use_lora:
        logger.info("LoRA requested: attempting to prepare LoRA adapters (requires 'peft' and 'bitsandbytes')")
        try:
            import peft
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            try:
                # bitsandbytes optional
                import bitsandbytes as bnb  # noqa: F401
                from bitsandbytes.optim import AdamW8bit
            except Exception:
                AdamW8bit = None

            # Robust search for underlying HF transformer model inside SentenceTransformer wrapper
            hf_model = None
            try:
                # 1) common attribute
                if hasattr(model, 'auto_model') and model.auto_model is not None:
                    hf_model = model.auto_model
                # 2) try _first_module() if available
                if hf_model is None and hasattr(model, '_first_module'):
                    try:
                        candidate = model._first_module()
                        if candidate is not None:
                            hf_model = candidate
                    except Exception:
                        pass
                # 3) search recursively for a submodule that looks like HF model
                if hf_model is None:
                    import transformers
                    for sub in model.modules():
                        try:
                            if isinstance(sub, transformers.PreTrainedModel):
                                hf_model = sub
                                break
                        except Exception:
                            # fallback: detect by presence of .config attribute
                            if hasattr(sub, 'config') and hasattr(sub, 'state_dict'):
                                hf_model = sub
                                break
            except Exception:
                hf_model = None

            if hf_model is None:
                logger.warning("Could not locate underlying HF transformer model inside SentenceTransformer. LoRA setup skipped.")
            else:
                # Prepare model for k-bit if bitsandbytes available
                try:
                    hf_model = prepare_model_for_kbit_training(hf_model)
                except Exception:
                    # prepare_model_for_kbit_training may not be necessary; continue
                    pass

                # Configure LoRA
                try:
                    # For embedding/inference models, FEATURE_EXTRACTION is correct
                    task_type = "FEATURE_EXTRACTION"
                    if hasattr(hf_model, 'config') and getattr(hf_model.config, 'is_decoder', False):
                        task_type = "CAUSAL_LM"
                except Exception:
                    task_type = "FEATURE_EXTRACTION"

                lora_config = LoraConfig(
                    r=effective_lora_r,
                    lora_alpha=effective_lora_alpha,
                    target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "fc1", "fc2"],
                    inference_mode=False,
                    bias="none",
                    task_type=task_type,
                )

                try:
                    hf_model = get_peft_model(hf_model, lora_config)
                    lora_applied = True
                    logger.info("LoRA adapters applied to underlying HF model")
                    # try to attach back to SentenceTransformer wrapper if possible
                    try:
                        if hasattr(model, 'auto_model'):
                            model.auto_model = hf_model
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Failed to apply LoRA adapters: {e}")
        except Exception as e:
            logger.warning(f"LoRA requested but 'peft' not available or setup failed: {e}")

    # Prepare a shuffled list of InputExample objects and iterate in micro-batches
    import random
    train_examples_list = list(train_examples)
    random.shuffle(train_examples_list)

    # Prefetch configuration: number of batches to keep ready (on CPU pinned memory)
    PREFETCH_BATCHES = 4
    PREFETCH_WORKERS = TRAINING_CONFIG.get('PREFETCH_WORKERS', 1)
    prefetch_queue = queue.Queue(maxsize=PREFETCH_BATCHES)

    # Helper: convert tokenized dict to pinned torch tensors (on CPU)
    def _to_pinned_tensor_dict(token_dict):
        pinned = {}
        for k, v in token_dict.items():
            if isinstance(v, torch.Tensor):
                # Ensure it's on CPU and pin it for faster GPU transfer
                t = v.detach().cpu()
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                pinned[k] = t
            else:
                pinned[k] = v
        return pinned

    # Producer thread: tokenizes and enqueues pinned tensor batches
    stop_producer = threading.Event()

    # Use the underlying HF tokenizer for bulk tokenization for better performance
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(model._first_module().config._name_or_path if hasattr(model, '_first_module') and hasattr(model._first_module().config, '_name_or_path') else None)
    except Exception:
        hf_tokenizer = None

    def _producer_thread_fn(batches):
        try:
            for batch in batches:
                if stop_producer.is_set():
                    break
                anchors = [ex.texts[0] for ex in batch]
                positives = [ex.texts[1] for ex in batch]
                negatives = [ex.texts[2] for ex in batch]
                if hf_tokenizer is not None:
                    a_inputs = hf_tokenizer(anchors, padding=True, truncation=True, return_tensors='pt')
                    p_inputs = hf_tokenizer(positives, padding=True, truncation=True, return_tensors='pt')
                    n_inputs = hf_tokenizer(negatives, padding=True, truncation=True, return_tensors='pt')
                else:
                    a_inputs = model.tokenize(anchors)
                    p_inputs = model.tokenize(positives)
                    n_inputs = model.tokenize(negatives)
                # Convert to pinned CPU tensors for faster transfer
                a_inputs = _to_pinned_tensor_dict(a_inputs)
                p_inputs = _to_pinned_tensor_dict(p_inputs)
                n_inputs = _to_pinned_tensor_dict(n_inputs)
                prefetch_queue.put((a_inputs, p_inputs, n_inputs))
        except Exception as e:
            logger.warning(f"Prefetch producer error: {e}")

    # Create list of micro-batches to feed the producer
    micro_batches = [train_examples_list[i:i + batch_size] for i in range(0, len(train_examples_list), batch_size)]
    # Start producer thread
    producer = threading.Thread(target=_producer_thread_fn, args=(micro_batches,), daemon=True)
    producer.start()

    # Reset CUDA peak memory tracking if available
    if device is not None and device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    try:
        # If LoRA applied, only optimize trainable parameters (adapters)
        if lora_applied:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            logger.info(f"LoRA applied: optimizing {len(trainable_params)} trainable parameters")
            # Prefer AdamW8bit if bitsandbytes is available
            if AdamW8bit is not None:
                try:
                    optimizer = AdamW8bit(trainable_params, lr=5e-6)
                    logger.info("Using AdamW8bit (bitsandbytes) for optimizer")
                except Exception as e:
                    logger.warning(f"AdamW8bit failed to initialize, falling back to AdamW: {e}")
                    optimizer = AdamW(trainable_params, lr=5e-6)
            else:
                optimizer = AdamW(trainable_params, lr=5e-6)
        else:
            optimizer = AdamW(model.parameters(), lr=5e-6)
    except Exception as e:
        # Catch allocator issues when creating optimizer state on GPU
        logger.warning(f"Optimizer creation failed: {e}")
        msg = str(e).lower()
        if ('out of memory' in msg) or isinstance(e, torch.cuda.OutOfMemoryError):
            # If config says to fail on CPU fallback, abort now
            if TRAINING_CONFIG.get('FAIL_ON_CPU_FALLBACK', False):
                logger.error("CUDA out of memory during optimizer creation and FAIL_ON_CPU_FALLBACK=True — aborting.")
                raise RuntimeError("Abort: CUDA OOM and CPU fallback disabled by configuration.")
            # Fallback to CPU once
            if not tried_fallback:
                logger.warning("CUDA out of memory during optimizer creation. Falling back to CPU and retrying the limited run.")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return run_limited_training_steps(model, train_examples, steps=steps, device=torch.device('cpu'), batch_size=1, grad_accum=1, use_fp16=False, tried_fallback=True)
        raise
    model.train()
    global_step = 0
    step_times = []

    use_amp = use_fp16 and (device is not None and device.type == 'cuda')
    # Use the new torch.amp API to avoid deprecation warnings
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    # Timeout controls from config
    max_step_seconds = TRAINING_CONFIG.get('MAX_STEP_SECONDS', None)
    max_total_seconds = TRAINING_CONFIG.get('MAX_TOTAL_SECONDS', None)
    run_start_time = time.time()

    try:
        while global_step < steps:
            # Get next pre-tokenized batch from the prefetch queue (blocks if empty)
            try:
                a_inputs, p_inputs, n_inputs = prefetch_queue.get(timeout=30)
            except Exception:
                logger.warning("Prefetch queue empty or timeout; falling back to on-demand tokenization")
                # On fallback, tokenize inline
                batch_idx = (global_step * batch_size) % len(train_examples_list)
                batch = train_examples_list[batch_idx:batch_idx + batch_size]
                anchors = [ex.texts[0] for ex in batch]
                positives = [ex.texts[1] for ex in batch]
                negatives = [ex.texts[2] for ex in batch]
                a_inputs = model.tokenize(anchors)
                p_inputs = model.tokenize(positives)
                n_inputs = model.tokenize(negatives)

            start = time.time()

            # Move tokenized input tensors (pinned CPU) to the same device as the model
            def _move_inputs(inp, device):
                if isinstance(inp, dict):
                    moved = {}
                    for k, v in inp.items():
                        if isinstance(v, torch.Tensor):
                            try:
                                moved[k] = v.to(device, non_blocking=True)
                            except Exception:
                                moved[k] = v.to(device)
                        else:
                            moved[k] = v
                    return moved
                return inp

            a_inputs = _move_inputs(a_inputs, device)
            p_inputs = _move_inputs(p_inputs, device)
            n_inputs = _move_inputs(n_inputs, device)

            if use_amp:
                # Use torch.amp.autocast with explicit device type to avoid deprecation warnings
                with torch.amp.autocast('cuda'):
                    a_emb = model(a_inputs)['sentence_embedding']
                    p_emb = model(p_inputs)['sentence_embedding']
                    n_emb = model(n_inputs)['sentence_embedding']
                    # Diagnostic checks for NaN/Inf
                    try:
                        # Emit DIAG logs only every N steps to reduce CPU/logging overhead
                        DIAG_EVERY_N = TRAINING_CONFIG.get('DIAG_EVERY_N_STEPS', 1)
                        def _diag(t, name):
                            if not isinstance(t, torch.Tensor):
                                return
                            logger.info(f"DIAG {name}: finite={torch.isfinite(t).all().item()}, any_nan={torch.isnan(t).any().item()}, mean={t.mean().item():.6f}, std={t.std().item():.6f}, min={t.min().item():.6f}, max={t.max().item():.6f}")
                        if ((global_step + 1) % DIAG_EVERY_N) == 0:
                            _diag(a_emb, 'a_emb')
                            _diag(p_emb, 'p_emb')
                            _diag(n_emb, 'n_emb')
                    except Exception as e:
                        logger.warning(f"Diagnostic check failed: {e}")
                    # If forward produced NaNs when using AMP, retry once without fp16
                    any_nan = False
                    try:
                        any_nan = (torch.isnan(a_emb).any().item() or torch.isnan(p_emb).any().item() or torch.isnan(n_emb).any().item())
                    except Exception:
                        any_nan = False
                    if any_nan and use_amp and not tried_fallback:
                        logger.warning("NaNs detected in embeddings during fp16 forward. Retrying limited run with fp16 disabled.")
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        return run_limited_training_steps(model, train_examples, steps=steps, device=device, batch_size=batch_size, grad_accum=grad_accum, use_fp16=False, tried_fallback=True, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
                    loss = torch.nn.functional.triplet_margin_loss(a_emb, p_emb, n_emb, margin=1.0)
            else:
                a_emb = model(a_inputs)['sentence_embedding']
                p_emb = model(p_inputs)['sentence_embedding']
                n_emb = model(n_inputs)['sentence_embedding']
                # Diagnostic checks for NaN/Inf
                try:
                    def _diag(t, name):
                        if not isinstance(t, torch.Tensor):
                            return
                        logger.info(f"DIAG {name}: finite={torch.isfinite(t).all().item()}, any_nan={torch.isnan(t).any().item()}, mean={t.mean().item():.6f}, std={t.std().item():.6f}, min={t.min().item():.6f}, max={t.max().item():.6f}")
                    _diag(a_emb, 'a_emb')
                    _diag(p_emb, 'p_emb')
                    _diag(n_emb, 'n_emb')
                except Exception as e:
                    logger.warning(f"Diagnostic check failed: {e}")
                loss = torch.nn.functional.triplet_margin_loss(a_emb, p_emb, n_emb, margin=1.0)
            loss = loss / grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % grad_accum == 0:
                try:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                except RuntimeError as e:
                    # Common cause: CUDA OOM when allocating optimizer state
                    msg = str(e).lower()
                    if 'out of memory' in msg:
                        logger.warning(f"CUDA OOM during optimizer.step(): {e}")
                        # Honor config: if user requested to fail on CPU fallback, abort here
                        if TRAINING_CONFIG.get('FAIL_ON_CPU_FALLBACK', False):
                            logger.error("CUDA OOM during optimizer.step() and FAIL_ON_CPU_FALLBACK=True — aborting.")
                            raise RuntimeError("Abort: CUDA OOM and CPU fallback disabled by configuration.")
                        if not tried_fallback:
                            logger.info("Attempting fallback: move model to CPU and retry limited run with smaller batch/use_fp16 disabled.")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            return run_limited_training_steps(model, train_examples, steps=steps-global_step, device=torch.device('cpu'), batch_size=1, grad_accum=1, use_fp16=False, tried_fallback=True)
                        else:
                            raise
                    else:
                        raise

            step_time = time.time() - start
            # Enforce per-step timeout
            if max_step_seconds and step_time > max_step_seconds:
                logger.error(f"Step exceeded MAX_STEP_SECONDS={max_step_seconds}s (took {step_time:.2f}s). Aborting.")
                raise RuntimeError("Abort: step exceeded max allowed duration")

            # Enforce total elapsed timeout
            if max_total_seconds and (time.time() - run_start_time) > max_total_seconds:
                logger.error(f"Total run exceeded MAX_TOTAL_SECONDS={max_total_seconds}s. Aborting.")
                raise RuntimeError("Abort: total run exceeded max allowed duration")
            step_times.append(step_time)

            # peak memory (if cuda)
            peak_mem = torch.cuda.max_memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0

            logger.info(f"step={global_step+1}, loss={loss.item():.6f}, time={step_time:.4f}s, peak_mem_mb={peak_mem:.1f}")

            global_step += 1
            if global_step >= steps:
                break

    except RuntimeError as e:
        # Catch any unexpected CUDA OOMs that slipped through
        msg = str(e).lower()
        if ('out of memory' in msg or isinstance(e, torch.cuda.OutOfMemoryError)):
            logger.warning(f"CUDA OOM during limited run: {e}")
            # Honor config: abort if CPU fallback is disabled
            if TRAINING_CONFIG.get('FAIL_ON_CPU_FALLBACK', False):
                logger.error("CUDA OOM during limited run and FAIL_ON_CPU_FALLBACK=True — aborting.")
                raise
            if not tried_fallback:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return run_limited_training_steps(model, train_examples, steps=steps, device=torch.device('cpu'), batch_size=1, grad_accum=1, use_fp16=False, tried_fallback=True)
        raise

    avg_time = sum(step_times) / len(step_times) if step_times else 0
    logger.info(f"Limited run complete: avg_time_per_step={avg_time:.4f}s, total_steps={global_step}")
    # Signal producer thread to stop and join
    try:
        stop_producer.set()
        if 'producer' in globals() and producer.is_alive():
            producer.join(timeout=2)
    except Exception:
        pass
    # Close profiler if enabled
    try:
        if 'profiler' in locals() and profiler is not None:
            try:
                profiler.__exit__(None, None, None)
                # Optionally export a trace file for analysis
                trace_path = TRAINING_CONFIG.get('TORCH_PROFILER_TRACE', None)
                if trace_path:
                    try:
                        profiler.export_chrome_trace(trace_path)
                        logger.info(f"torch.profiler trace exported to {trace_path}")
                    except Exception as e:
                        logger.warning(f"Failed to export torch.profiler trace: {e}")
            except Exception:
                pass
    except Exception:
        pass
    return {
        'steps': global_step,
        'avg_time_per_step': avg_time,
        'peak_mem_mb': peak_mem,
    }