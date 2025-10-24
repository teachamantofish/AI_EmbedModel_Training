import torch
import time
from embedding_finetuner import setup_model, load_training_data
from config_embed_training import TRAINING_CONFIG

OUT_TRACE = 'trace_forward.json'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = setup_model(device=None)
    model = model.to(device)
    model.eval()

    train_examples, _ = load_training_data()
    batch = train_examples[:1]
    anchors = [ex.texts[0] for ex in batch]
    p = [ex.texts[1] for ex in batch]
    n = [ex.texts[2] for ex in batch]

    a_inputs = model.tokenize(anchors)
    p_inputs = model.tokenize(p)
    n_inputs = model.tokenize(n)
    a_inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in a_inputs.items()}
    p_inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in p_inputs.items()}
    n_inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in n_inputs.items()}

    use_fp16 = TRAINING_CONFIG.get('fp16', False)

    try:
        from torch import profiler
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
        with profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.no_grad():
                if use_fp16 and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        _ = model(a_inputs)['sentence_embedding']
                else:
                    _ = model(a_inputs)['sentence_embedding']
        # export trace
        try:
            prof.export_chrome_trace(OUT_TRACE)
            print('Trace exported to', OUT_TRACE)
        except Exception as e:
            print('Could not export trace:', e)
        # show top ops
        try:
            print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
        except Exception as e:
            print('Could not print profiler table:', e)
    except Exception as e:
        print('Profiler not available or failed:', e)

if __name__ == '__main__':
    main()
