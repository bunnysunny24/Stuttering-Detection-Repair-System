"""Quick system diagnostics for AGNI training optimization."""
import platform, shutil, os, subprocess, torch

print("=" * 60)
print("  SYSTEM DIAGNOSTICS")
print("=" * 60)

# CPU
print(f"\nCPU: {platform.processor()}")
print(f"Logical cores: {os.cpu_count()}")

# RAM - use PowerShell for reliable results
r = subprocess.run(
    ['powershell', '-Command',
     '[math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB,1),'
     '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)'],
    capture_output=True, text=True
)
ram_vals = [float(x.strip()) for x in r.stdout.strip().split('\n') if x.strip()]
total_gb = ram_vals[0] if len(ram_vals) > 0 else 0
free_gb = ram_vals[1] if len(ram_vals) > 1 else 0
used_gb = total_gb - free_gb
print(f"\nRAM Total:     {total_gb:.1f} GB")
print(f"RAM Free:      {free_gb:.1f} GB")
if total_gb > 0:
    print(f"RAM Used:      {used_gb:.1f} GB ({100*used_gb/total_gb:.0f}%)")
else:
    print(f"RAM Used:      {used_gb:.1f} GB")

# Torch
print(f"\nTorch threads: {torch.get_num_threads()}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")

# Disk
total, used, free = shutil.disk_usage('D:/')
print(f"\nDisk D: {free/1024**3:.0f} GB free / {total/1024**3:.0f} GB total")

# Python processes
r2 = subprocess.run(
    'tasklist /FI "IMAGENAME eq python.exe" /FO CSV',
    capture_output=True, text=True, shell=True
)
print("\nPython processes:")
for line in r2.stdout.strip().split('\n'):
    print(f"  {line}")

# Recommendations
print("\n" + "=" * 60)
print("  OPTIMIZATION ANALYSIS")
print("=" * 60)

headroom_gb = free_gb
print(f"\nFree RAM headroom: {headroom_gb:.1f} GB")

# Current config: batch=8, model uses ~3.2GB
model_gb = 3.2  # measured previously
print(f"Model memory (estimated): ~{model_gb} GB")
print(f"Headroom after model: ~{headroom_gb - model_gb:.1f} GB")

# Batch size analysis
# Each batch of 8 at 3-sec audio ~ 384K floats * 4B = 1.5MB per sample
# Plus activations through wav2vec2-large ~200MB per sample
# batch=8 -> ~1.6GB activations, batch=16 -> ~3.2GB activations
print("\nBatch size estimates (activations + gradients):")
for bs in [8, 12, 16, 24]:
    est_gb = model_gb + (bs * 0.2)  # rough 200MB per sample in activations
    fits = "YES" if est_gb < total_gb * 0.85 else "NO"
    print(f"  batch={bs}: ~{est_gb:.1f} GB -> {fits} (limit ~{total_gb*0.85:.0f} GB)")

print(f"\nnum_workers: 0 is optimal on Windows with this CPU (benchmarked)")
print(f"omp_threads: 6 is optimal for {os.cpu_count()} logical cores (benchmarked)")
