import time
import subprocess
import torch

def nvidia_smi_snapshot():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ])
        return out.decode().strip()
    except Exception as e:
        return f'nvidia-smi not available: {e}'


def try_matmul(N, iterations=3):
    print(f"\n=== MATMUL N={N}, iterations={iterations} ===")
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        # warmup
        for _ in range(1):
            A = torch.randn((N, N), device='cuda', dtype=torch.float32)
            B = torch.randn((N, N), device='cuda', dtype=torch.float32)
            C = A @ B
            torch.cuda.synchronize()
            del A, B, C
            torch.cuda.empty_cache()

        for i in range(iterations):
            A = torch.randn((N, N), device='cuda', dtype=torch.float32)
            B = torch.randn((N, N), device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            t0 = time.time()
            C = A @ B
            torch.cuda.synchronize()
            t1 = time.time()
            elapsed = t1 - t0
            times.append(elapsed)
            print(f"iter {i+1}/{iterations}: {elapsed:.4f} s")
            # free
            del A, B, C
            torch.cuda.empty_cache()
        peak_mem = torch.cuda.max_memory_reserved() / (1024**2)
        avg = sum(times) / len(times)
        flops = 2.0 * (N**3)  # FLOPs for matmul
        gflops = (flops / avg) / 1e9
        print(f"N={N} avg time: {avg:.4f} s, GFLOPS: {gflops:.1f}, peak_mem(MB): {peak_mem:.1f}")
        print('nvidia-smi snapshot:', nvidia_smi_snapshot())
        return True
    except RuntimeError as e:
        print(f"RuntimeError for N={N}: {e}")
        return False
    except Exception as e:
        print(f"Error for N={N}: {e}")
        return False


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA not available in this Python environment')
        print('torch version:', torch.__version__)
        raise SystemExit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}, total_memory_MB: {props.total_memory/1024**2:.0f}")
    print('torch.version.cuda', torch.version.cuda)
    print('nvidia-smi snapshot before test:', nvidia_smi_snapshot())

    sizes = [16000, 12000, 10000, 8192, 6000]
    for N in sizes:
        ok = try_matmul(N, iterations=3)
        if ok:
            # stop after first successful heavy run
            break
    print('\nDone')
