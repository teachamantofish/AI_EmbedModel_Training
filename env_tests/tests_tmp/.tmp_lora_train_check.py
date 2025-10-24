import time
import torch
from embedding_finetuner import setup_model, load_training_data
from config_embed_training import TRAINING_CONFIG

STEPS = 2
BATCH_SIZE = 1
USE_FP16 = TRAINING_CONFIG.get('fp16', False)


def to_device(inp, device):
    if isinstance(inp, dict):
        moved = {}
        for k, v in inp.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device, non_blocking=True)
            else:
                moved[k] = v
        return moved
    return inp


def apply_lora_if_possible(model):
    try:
        import peft
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        # find hf_model inside SentenceTransformer
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
        if hf_model is None:
            # search
            try:
                import transformers
                for sub in model.modules():
                    try:
                        if isinstance(sub, transformers.PreTrainedModel):
                            hf_model = sub
                            break
                    except Exception:
                        if hasattr(sub, 'config') and hasattr(sub, 'state_dict'):
                            hf_model = sub
                            break
            except Exception:
                hf_model = None
        if hf_model is None:
            print('Could not locate HF model for LoRA')
            return model, False
        # try to prepare for kbit
        try:
            hf_model = prepare_model_for_kbit_training(hf_model)
        except Exception:
            pass
        lora_config = LoraConfig(
            r=TRAINING_CONFIG.get('lora_r', 8),
            lora_alpha=TRAINING_CONFIG.get('lora_alpha', 16),
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "fc1", "fc2"],
            inference_mode=False,
            bias='none',
            task_type='FEATURE_EXTRACTION',
        )
        hf_model = get_peft_model(hf_model, lora_config)
        # attach back
        try:
            if hasattr(model, 'auto_model'):
                model.auto_model = hf_model
        except Exception:
            pass
        print('LoRA adapters applied')
        return model, True
    except Exception as e:
        print('PEFT/LoRA not available or failed:', e)
        return model, False


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = setup_model(device=None)
    model = model.to(device)

    # apply LoRA if possible (reduces number of trainable parameters)
    model, lora_applied = apply_lora_if_possible(model)

    train_examples, _ = load_training_data()
    samples = train_examples[: max(BATCH_SIZE * STEPS, 8)]

    # collect trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print('trainable_params_count:', len(trainable_params))

    # choose optimizer
    AdamW8bit = None
    try:
        import bitsandbytes as bnb
        from bitsandbytes.optim import AdamW8bit as _AdamW8bit
        AdamW8bit = _AdamW8bit
        print('bitsandbytes AdamW8bit available')
    except Exception:
        print('bitsandbytes not available')

    try:
        if lora_applied and AdamW8bit is not None and trainable_params:
            optimizer = AdamW8bit(trainable_params, lr=5e-6)
            opt_name = 'AdamW8bit'
        elif trainable_params:
            optimizer = torch.optim.AdamW(trainable_params, lr=5e-6)
            opt_name = 'AdamW'
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
            opt_name = 'AdamW_full'
        print('Optimizer:', opt_name)
    except Exception as e:
        print('Optimizer init failed:', e)
        return

    scaler = torch.amp.GradScaler('cuda') if (USE_FP16 and device.type == 'cuda') else None

    token_times = []
    h2d_times = []
    fwd_bwd_times = []
    opt_times = []

    for step in range(STEPS):
        batch = samples[(step * BATCH_SIZE) % len(samples) : ((step + 1) * BATCH_SIZE) % len(samples)]
        if not batch:
            batch = samples[:BATCH_SIZE]
        anchors = [ex.texts[0] for ex in batch]
        positives = [ex.texts[1] for ex in batch]
        negatives = [ex.texts[2] for ex in batch]

        t0 = time.time()
        a_inputs = model.tokenize(anchors)
        p_inputs = model.tokenize(positives)
        n_inputs = model.tokenize(negatives)
        t1 = time.time()
        token_times.append(t1 - t0)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()
        a_inputs = to_device(a_inputs, device)
        p_inputs = to_device(p_inputs, device)
        n_inputs = to_device(n_inputs, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.time()
        h2d_times.append(t3 - t2)

        # forward + backward
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            t4 = time.time()
            if USE_FP16 and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    a_emb = model(a_inputs)['sentence_embedding']
                    p_emb = model(p_inputs)['sentence_embedding']
                    n_emb = model(n_inputs)['sentence_embedding']
                    loss = torch.nn.functional.triplet_margin_loss(a_emb, p_emb, n_emb, margin=1.0)
                scaler.scale(loss).backward()
            else:
                a_emb = model(a_inputs)['sentence_embedding']
                p_emb = model(p_inputs)['sentence_embedding']
                n_emb = model(n_inputs)['sentence_embedding']
                loss = torch.nn.functional.triplet_margin_loss(a_emb, p_emb, n_emb, margin=1.0)
                loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t5 = time.time()
            fwd_bwd_times.append(t5 - t4)

            # optimizer step
            t6 = time.time()
            if USE_FP16 and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t7 = time.time()
            opt_times.append(t7 - t6)

            peak_mem = torch.cuda.max_memory_reserved() / (1024 * 1024) if device.type == 'cuda' else 0

            print(f"step={step+1}: token={token_times[-1]:.4f}s, h2d={h2d_times[-1]:.4f}s, fwd_bwd={fwd_bwd_times[-1]:.4f}s, opt={opt_times[-1]:.4f}s, peak_mem_mb={peak_mem:.1f}")

        except RuntimeError as e:
            print('RuntimeError during fwd/bwd/step:', e)
            return

    import statistics
    print('\nSUMMARY')
    print(f"avg_token={statistics.mean(token_times):.4f}s, avg_h2d={statistics.mean(h2d_times):.4f}s, avg_fwd_bwd={statistics.mean(fwd_bwd_times):.4f}s, avg_opt={statistics.mean(opt_times):.4f}s")

if __name__ == '__main__':
    main()
