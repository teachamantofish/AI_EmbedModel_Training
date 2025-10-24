import time
import torch
import subprocess

def nvidia_smi():
    try:
        out = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'])
        return out.decode().strip()
    except Exception as e:
        return f'nvidia-smi error: {e}'

if __name__ == '__main__':
    target_seconds = 15.0
    N = 16000
    iterations = 0
    times = []

    if not torch.cuda.is_available():
        print('CUDA not available')
        raise SystemExit(1)

    print('Device:', torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()
    start_all = time.time()
    while True:
        try:
            A = torch.randn((N, N), device='cuda', dtype=torch.float32)
            B = torch.randn((N, N), device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            t0 = time.time()
            C = A @ B
            torch.cuda.synchronize()
            t1 = time.time()
            elapsed = t1 - t0
            times.append(elapsed)
            iterations += 1
            # free some memory
            del A, B, C
            torch.cuda.empty_cache()
            total_elapsed = time.time() - start_all
            print(f'iter {iterations}: {elapsed:.4f}s, total {total_elapsed:.1f}s')
            if total_elapsed >= target_seconds:
                break
        except RuntimeError as e:
            print('RuntimeError:', e)
            break

    avg = sum(times)/len(times) if times else 0
    flops = 2.0 * (N**3)
    gflops = (flops / avg) / 1e9 if avg>0 else 0
    peak_mem = torch.cuda.max_memory_reserved()/(1024**2)
    print('--- RESULT ---')
    print('iterations', iterations)
    print(f'avg_time {avg:.4f}s, GFLOPS {gflops:.1f}, peak_mem_MB {peak_mem:.1f}')
    print('nvidia-smi snapshot:', nvidia_smi())
