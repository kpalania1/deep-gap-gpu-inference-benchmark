#!/usr/bin/env python3
import csv
import os
import platform
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
from torchvision import models
import pynvml

# Enable cuDNN autotuning
torch.backends.cudnn.benchmark = True

CSV_FILE = "benchmark_results_final.csv"

# ==============================
# Parameters
# ==============================
BATCH_SIZES  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
WARMUP_ITERS = 20
TIMED_ITERS  = 100
SWEEPS_S     = 10
REPEATS_R    = 3

torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"

# ==============================
# NVML total GPU memory used
# ==============================
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def try_get_total_used_mem_mb_via_nvml():
    """
    Return total GPU memory currently used on device 0, in MB.
    This is total system GPU memory usage reported by NVML.
    """
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        return round(info.used / (1024 * 1024), 1)
    except Exception as e:
        print(f"[WARN] NVML read failed: {e}")
        return None


# ==============================
# Sysinfo helpers
# ==============================
def sh(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def parse_lscpu() -> dict:
    info = {}
    txt = sh("lscpu")
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def parse_meminfo() -> dict:
    txt = sh("free -m")
    total_mb = ""
    for line in txt.splitlines():
        if line.lower().startswith("mem:"):
            parts = line.split()
            if len(parts) >= 2:
                total_mb = parts[1]
    return {"mem_total_mb": total_mb}


lscpu = parse_lscpu()
mem = parse_meminfo()

sysinfo = {
    "hostname": sh("hostname"),
    "os": sh("cat /etc/os-release | egrep '^(NAME|VERSION)=' | tr '\n' ' '"),
    "kernel": platform.release(),
    "arch": platform.machine(),
    "cpu_model": lscpu.get("Model name", ""),
    "cpu_sockets": lscpu.get("Socket(s)", ""),
    "cpu_cores_per_socket": lscpu.get("Core(s) per socket", ""),
    "cpu_threads_per_core": lscpu.get("Thread(s) per core", ""),
    "cpu_logical_cpus": lscpu.get("CPU(s)", ""),
    "mem_total_mb": mem.get("mem_total_mb", ""),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torchvision_version": torchvision.__version__,
}

# ==============================
# CSV Header
# ==============================
header = [
    "timestamp_utc", "device", "sweep_id", "repeat_id", "model", "batch_size",
    "threads", "iters", "warmup_iters",
    "median_ms", "mean_ms", "std_latency_ms", "p99_latency_ms",
    "throughput_ips",
    "hostname", "os", "kernel", "arch", "cpu_model",
    "cpu_sockets", "cpu_cores_per_socket", "cpu_threads_per_core",
    "cpu_logical_cpus", "mem_total_mb", "python_version", "torch_version",
    "cuda_available", "torchvision_version", "gpu_peak_mem_mb"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

# ==============================
# Device
# ==============================
use_cuda = torch.cuda.is_available()
device_name = "cuda" if use_cuda else "cpu"
device = torch.device(device_name)

if use_cuda:
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA version: {torch.version.cuda}")

# ==============================
# Models
# ==============================
model_builders = [
    ("resnet101", lambda: models.resnet101(pretrained=True)),
    ("resnet50", lambda: models.resnet50(pretrained=True)),
    ("resnet18", lambda: models.resnet18(pretrained=True)),
]

# ==============================
# Threads
# ==============================
logical_cpus = int(sysinfo["cpu_logical_cpus"]) if sysinfo["cpu_logical_cpus"] else 1
THREADS_LIST = [1] if use_cuda else [t for t in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48] if t <= logical_cpus]

# ==============================
# Timing
# ==============================
def timed_inference_ms(model, x):
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        _ = model(x)
        ender.record()

        torch.cuda.synchronize()
        return float(starter.elapsed_time(ender))
    else:
        start = time.perf_counter()
        _ = model(x)
        return (time.perf_counter() - start) * 1000.0


def is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg


# ==============================
# Benchmark
# ==============================
print(f"Logging to: {CSV_FILE} (Updated in real-time)")
print("-" * 140)

for model_name, builder in model_builders:
    model = builder().to(device).half()
    model.eval()

    with torch.inference_mode():
        for s in range(1, SWEEPS_S + 1):
            for t in THREADS_LIST:
                if not use_cuda:
                    torch.set_num_threads(t)

                for b in BATCH_SIZES:
                    try:
                        x = torch.randn(b, 3, 224, 224, device=device, dtype=torch.float16)
                    except RuntimeError as e:
                        if use_cuda and is_oom(e):
                            print(f"[SKIP] {model_name} | B:{b} -> OOM alloc")
                            torch.cuda.empty_cache()
                            continue
                        raise

                    pending_rows = []
                    batch_max_gpu_peak_mem_mb = 0.0

                    for r in range(1, REPEATS_R + 1):
                        max_mem_mb = 0.0

                        # Warmup
                        for _ in range(WARMUP_ITERS):
                            _ = model(x)
                            if use_cuda:
                                torch.cuda.synchronize()
                                mem_mb = try_get_total_used_mem_mb_via_nvml()
                                if mem_mb is not None and mem_mb > max_mem_mb:
                                    max_mem_mb = mem_mb

                        if use_cuda:
                            torch.cuda.synchronize()
                            mem_mb = try_get_total_used_mem_mb_via_nvml()
                            if mem_mb is not None and mem_mb > max_mem_mb:
                                max_mem_mb = mem_mb

                        latencies = []

                        for _ in range(TIMED_ITERS):
                            lat_ms = timed_inference_ms(model, x)
                            latencies.append(lat_ms)

                            if use_cuda:
                                mem_mb = try_get_total_used_mem_mb_via_nvml()
                                if mem_mb is not None and mem_mb > max_mem_mb:
                                    max_mem_mb = mem_mb

                        if max_mem_mb > batch_max_gpu_peak_mem_mb:
                            batch_max_gpu_peak_mem_mb = max_mem_mb

                        median_ms = float(np.median(latencies))
                        mean_ms = float(np.mean(latencies))
                        std_ms = float(np.std(latencies))
                        p99_ms = float(np.percentile(latencies, 99))
                        ips = b / (median_ms / 1000.0)

                        pending_rows.append([
                            datetime.utcnow().isoformat(),
                            device_name,
                            s,
                            r,
                            model_name,
                            b,
                            t,
                            TIMED_ITERS,
                            WARMUP_ITERS,
                            round(median_ms, 3),
                            round(mean_ms, 3),
                            round(std_ms, 3),
                            round(p99_ms, 3),
                            round(ips, 3),
                            sysinfo["hostname"],
                            sysinfo["os"],
                            sysinfo["kernel"],
                            sysinfo["arch"],
                            sysinfo["cpu_model"],
                            sysinfo["cpu_sockets"],
                            sysinfo["cpu_cores_per_socket"],
                            sysinfo["cpu_threads_per_core"],
                            sysinfo["cpu_logical_cpus"],
                            sysinfo["mem_total_mb"],
                            sysinfo["python_version"],
                            sysinfo["torch_version"],
                            sysinfo["cuda_available"],
                            sysinfo["torchvision_version"],
                            None,  # fill batch-level memory later
                        ])

                        mem_str = f" | PeakUsedMem(maxR): {max_mem_mb:>6.1f} MB" if use_cuda and max_mem_mb > 0 else ""

                        print(
                            f"{model_name:<10} | S:{s:02d} | R:{r} | T:{t:02d} | B:{b:<3} | "
                            f"Med: {median_ms:10.2f} ms | "
                            f"P99: {p99_ms:10.2f} ms | "
                            f"{ips:8.2f} ips{mem_str}"
                        )

                    # Final batch-level max across repeats
                    final_gpu_peak_mem_mb = round(batch_max_gpu_peak_mem_mb, 1) if use_cuda and batch_max_gpu_peak_mem_mb > 0 else ""

                    for row in pending_rows:
                        row[-1] = final_gpu_peak_mem_mb

                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(pending_rows)

print("-" * 140)
print("Benchmark Complete.")
