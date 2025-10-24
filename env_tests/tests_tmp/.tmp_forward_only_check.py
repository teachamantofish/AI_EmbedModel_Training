import time
import torch
from embedding_finetuner import setup_model, load_training_data
from config_embed_training import TRAINING_CONFIG

STEPS = 6
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print('Loading model (local path) ...')
    model = setup_model(device=None)
    model = model.to(device)
    model.eval()

    train_examples, _ = load_training_data()
    if len(train_examples) == 0:
        print('No training examples found. Abort.')
        return

    samples = train_examples[: max(BATCH_SIZE * STEPS, 8)]

    token_times = []
    h2d_times = []
    fwd_times = []

    for step in range(STEPS):
        batch = samples[(step * BATCH_SIZE) % len(samples) : ((step + 1) * BATCH_SIZE) % len(samples)]
        if not batch:
            batch = samples[:BATCH_SIZE]

        anchors = [ex.texts[0] for ex in batch]
        positives = [ex.texts[1] for ex in batch]
        negatives = [ex.texts[2] for ex in batch]

        # Tokenize on CPU
        t0 = time.time()
        try:
            a_inputs = model.tokenize(anchors)
            p_inputs = model.tokenize(positives)
            n_inputs = model.tokenize(negatives)
        except Exception as e:
            print('model.tokenize failed:', e)
            return
        t1 = time.time()
        token_dur = t1 - t0

        # Host -> device
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()
        a_inputs_dev = to_device(a_inputs, device)
        p_inputs_dev = to_device(p_inputs, device)
        n_inputs_dev = to_device(n_inputs, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.time()
        h2d_dur = t3 - t2

        # Forward pass only (no backward, no optimizer)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        if USE_FP16 and device.type == 'cuda':
            t4 = time.time()
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    a_emb = model(a_inputs_dev)['sentence_embedding']
                    p_emb = model(p_inputs_dev)['sentence_embedding']
                    n_emb = model(n_inputs_dev)['sentence_embedding']
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t5 = time.time()
            fwd_dur = t5 - t4
        else:
            t4 = time.time()
            with torch.no_grad():
                a_emb = model(a_inputs_dev)['sentence_embedding']
                p_emb = model(p_inputs_dev)['sentence_embedding']
                n_emb = model(n_inputs_dev)['sentence_embedding']
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t5 = time.time()
            fwd_dur = t5 - t4

        peak_mem = torch.cuda.max_memory_reserved() / (1024 * 1024) if device.type == 'cuda' else 0

        token_times.append(token_dur)
        h2d_times.append(h2d_dur)
        fwd_times.append(fwd_dur)

        print(f"step={step+1}: token={token_dur:.4f}s, h2d={h2d_dur:.4f}s, forward={fwd_dur:.4f}s, peak_mem_mb={peak_mem:.1f}")

    import statistics
    print('\nSUMMARY (avg over steps)')
    print(f"avg_token_time = {statistics.mean(token_times):.4f}s")
    print(f"avg_host_to_device = {statistics.mean(h2d_times):.4f}s")
    print(f"avg_forward = {statistics.mean(fwd_times):.4f}s")

if __name__ == '__main__':
    main()
