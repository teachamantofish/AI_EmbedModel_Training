import os
import torch
from pathlib import Path
from config_embed_training import TRAINING_CONFIG, MODEL_NAME

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running diagnostic on device: {device}")

    # Load SentenceTransformer similar to setup_model behaviour
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(MODEL_NAME))
        # Move model directly to device to mimic training path
        try:
            model = model.to(device)
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Attempt to detect HF backbone
    hf_model = None
    try:
        if hasattr(model, 'auto_model') and model.auto_model is not None:
            hf_model = model.auto_model
        elif hasattr(model, '_first_module'):
            try:
                hf_model = model._first_module()
            except Exception:
                hf_model = None
    except Exception:
        hf_model = None

    use_lora = TRAINING_CONFIG.get('use_lora', False)
    AdamW8bit = None
    if use_lora:
        print('LoRA requested; attempting to apply adapters (best-effort)')
        try:
            import peft
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            # locate hf_model again more thoroughly
            if hf_model is None:
                for sub in model.modules():
                    try:
                        if hasattr(sub, 'config') and hasattr(sub, 'state_dict'):
                            hf_model = sub
                            break
                    except Exception:
                        continue
            if hf_model is None:
                print('Could not locate HF model inside SentenceTransformer wrapper; skipping LoRA')
            else:
                try:
                    hf_model = prepare_model_for_kbit_training(hf_model)
                except Exception:
                    pass
                lora_cfg = LoraConfig(r=TRAINING_CONFIG.get('lora_r', 8), lora_alpha=TRAINING_CONFIG.get('lora_alpha', 16), target_modules=["q_proj","v_proj","o_proj","k_proj","fc1","fc2"], inference_mode=False, bias='none', task_type='FEATURE_EXTRACTION')
                try:
                    hf_model = get_peft_model(hf_model, lora_cfg)
                    # attach back
                    try:
                        if hasattr(model, 'auto_model'):
                            model.auto_model = hf_model
                    except Exception:
                        pass
                    print('LoRA adapters applied to HF model')
                except Exception as e:
                    print(f'Failed to apply LoRA adapters: {e}')
        except Exception as e:
            print(f'PEFT/LoRA not available: {e}')

    # Detect bitsandbytes optimizer availability
    try:
        import bitsandbytes as bnb
        from bitsandbytes.optim import AdamW8bit as _AdamW8bit
        AdamW8bit = _AdamW8bit
        print('bitsandbytes detected; AdamW8bit available')
    except Exception:
        print('bitsandbytes not available')

    # Build optimizer similarly to training code
    try:
        from torch.optim import AdamW
        # If LoRA applied, only opt trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f'Trainable parameters: {sum(p.numel() for p in trainable)} ({len(trainable)} tensors)')
        if TRAINING_CONFIG.get('use_lora', False) and AdamW8bit is not None:
            try:
                opt = AdamW8bit(trainable, lr=TRAINING_CONFIG.get('learning_rate', 5e-6))
                print('Initialized AdamW8bit optimizer (bitsandbytes)')
            except Exception as e:
                print(f'AdamW8bit init failed: {e}; falling back to AdamW')
                opt = AdamW(trainable, lr=TRAINING_CONFIG.get('learning_rate', 5e-6))
        else:
            opt = AdamW(trainable if TRAINING_CONFIG.get('use_lora', False) else model.parameters(), lr=TRAINING_CONFIG.get('learning_rate', 5e-6))
            print(f'Initialized optimizer: {opt.__class__.__name__}')
    except Exception as e:
        print(f'Failed to create optimizer: {e}')

    # Print CUDA memory summary
    if device.type == 'cuda':
        try:
            print('\n--- torch.cuda.memory_summary() ---')
            print(torch.cuda.memory_summary(device=device, abbreviated=False))
        except Exception as e:
            print(f'Could not get memory_summary: {e}')
    else:
        print('No CUDA device available to report memory summary')

if __name__ == '__main__':
    main()
