"""Estimate safe training batch sizes from local hardware.

Run inside your activated `agni` conda env:
  conda activate agni
  python scripts\estimate_batch_size.py

Outputs GPU memory info (via PyTorch) if available, otherwise RAM-based recommendation.
"""

import sys

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

def gb(x):
    return float(x) / (1024 ** 3)

def recommend_by_gpu(mem_gb):
    if mem_gb >= 24:
        return 64
    if mem_gb >= 12:
        return 32
    if mem_gb >= 8:
        return 16
    if mem_gb >= 6:
        return 8
    if mem_gb >= 4:
        return 4
    return 2

def recommend_by_ram(ram_gb):
    if ram_gb >= 128:
        return 64
    if ram_gb >= 64:
        return 32
    if ram_gb >= 32:
        return 16
    if ram_gb >= 16:
        return 8
    return 4

print("=== AGNI Batch Size Estimator ===")

# RAM
total_ram = None
if psutil:
    total_ram = psutil.virtual_memory().total
    print(f"Total RAM: {gb(total_ram):.1f} GB")
else:
    try:
        import os
        if sys.platform == 'win32':
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint),
                    ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram = stat.ullTotalPhys
            print(f"Total RAM: {gb(total_ram):.1f} GB")
    except Exception:
        print("Total RAM: unknown (install psutil for accurate detection)")

# GPU via torch
gpu_info = []
if torch is not None:
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                gpu_info.append({'idx': i, 'name': prop.name, 'mem_gb': prop.total_memory / 1024**3})
    except Exception:
        gpu_info = []

if gpu_info:
    for g in gpu_info:
        rec = recommend_by_gpu(g['mem_gb'])
        print(f"GPU[{g['idx']}]: {g['name']} — {g['mem_gb']:.1f} GB -> suggested per-step batch size: {rec}")
else:
    print("No CUDA GPU detected or PyTorch not available. Using RAM-based guidance.")
    if total_ram:
        rec = recommend_by_ram(gb(total_ram))
        print(f"RAM-based suggestion -> per-step batch size: {rec}")
    else:
        print("Install psutil or run inside conda env with psutil to get RAM detection.")

print('\nPractical tips:')
print('- Monitor GPU memory with `nvidia-smi` while testing one training step.')
print('- If OOM, halve the batch size and retry.')
print('- Use gradient accumulation to increase effective batch size without more memory (e.g., batch=8, accumulate=4 -> eff=32).')
print('- For CPU-only training, prefer smaller batches and reduce DataLoader workers.')
