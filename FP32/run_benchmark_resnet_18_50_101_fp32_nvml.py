import csv, os, platform, subprocess, time
from datetime import datetime
import numpy as np
import torch
import torchvision
from torchvision import models

torch.backends.cudnn.benchmark = True

# Final file name
CSV_FILE = "benchmark_results_final.csv"

# ---- Parameters ----
BATCH_SIZES  = [1,2,4,8,16,32,64,128,256,384,512]
WARMUP_ITERS = 20
TIMED_ITERS  = 100
SWEEPS_S     = 10
REPEATS_R    = 3

# Force single-threaded inter-op to ensure deterministic results
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"

# --- Sysinfo Helpers ---
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

# --- NVML Helper ---
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f"[WARN] NVML init failed: {e}")
    _NVML_HANDLE = None

def try_get_gpu_mem_mb_via_nvml() -> float | None:
    try:
        if _NVML_HANDLE is None:
            return None
        info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
        return round(info.used / (1024 * 1024), 1)
    except Exception as e:
        print(f"[WARN] NVML read failed: {e}")
        return None

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

# CSV Header
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

# Create file with header if missing
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

# --- Execution ---
use_cuda = torch.cuda.is_available()
device_name = "cuda" if use_cuda else "cpu"
device = torch.device(device_name)

if use_cuda:
    try:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version: {torch.version.cuda}")
    except Exception:
        pass

model_builders = [
    ("resnet101", lambda: models.resnet101(pretrained=True)),
    ("resnet50",  lambda: models.resnet50(pretrained=True)),
    ("resnet18",  lambda: models.resnet18(pretrained=True)),
]

# Safety check to prevent crashing on the smaller old server
logical_cpus = int(sysinfo["cpu_logical_cpus"]) if sysinfo["cpu_logical_cpus"] else 1
THREADS_LIST = [t for t in [1,2,3,4,6,8,12,16,24,32,40,48] if t <= logical_cpus]

if use_cuda:
    THREADS_LIST = [1]

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
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)

print(f"Logging to: {CSV_FILE} (Updated in real-time)")
print("-" * 140)

for model_name, builder in model_builders:
    model = builder().to(device)
    model.eval()

    with torch.inference_mode():
        for s in range(1, SWEEPS_S + 1):
            for t in THREADS_LIST:
                if not use_cuda:
                    torch.set_num_threads(t)

                for b in BATCH_SIZES:
                    try:
                        x = torch.randn(b, 3, 224, 224, device=device)
                    except RuntimeError as e:
                        if use_cuda and is_oom(e):
                            print(f"[SKIP] {model_name} | S:{s:02d} | T:{t:02d} | B:{b:<3} -> OOM alloc")
                            torch.cuda.empty_cache()
                            continue
                        raise

                    repeat_rows = []
                    batch_max_gpu_peak_mem_mb = 0.0
                    batch_had_valid_mem = False
                    skip_batch = False

                    for r in range(1, REPEATS_R + 1):
                        max_mem_mb = 0.0

                        # Warmup Phase
                        try:
                            for _ in range(WARMUP_ITERS):
                                _ = model(x)

                                if use_cuda:
                                    mem_mb = try_get_gpu_mem_mb_via_nvml()
                                    if mem_mb is not None and mem_mb > max_mem_mb:
                                        max_mem_mb = mem_mb

                            if use_cuda:
                                torch.cuda.synchronize()
                                mem_mb = try_get_gpu_mem_mb_via_nvml()
                                if mem_mb is not None and mem_mb > max_mem_mb:
                                    max_mem_mb = mem_mb

                        except RuntimeError as e:
                            if use_cuda and is_oom(e):
                                print(f"[SKIP] {model_name} | S:{s:02d} | R:{r} | B:{b:<3} -> OOM warmup")
                                torch.cuda.empty_cache()
                                skip_batch = True
                                break
                            raise

                        latencies = []

                        try:
                            for _ in range(TIMED_ITERS):
                                lat_ms = timed_inference_ms(model, x)
                                latencies.append(lat_ms)

                                if use_cuda:
                                    mem_mb = try_get_gpu_mem_mb_via_nvml()
                                    if mem_mb is not None and mem_mb > max_mem_mb:
                                        max_mem_mb = mem_mb

                        except RuntimeError as e:
                            if use_cuda and is_oom(e):
                                print(f"[SKIP] {model_name} | S:{s:02d} | R:{r} | B:{b:<3} -> OOM timing")
                                torch.cuda.empty_cache()
                                skip_batch = True
                                break
                            raise

                        repeat_peak_total_mb = None
                        if use_cuda and max_mem_mb > 0:
                            repeat_peak_total_mb = round(max_mem_mb, 1)
                            batch_max_gpu_peak_mem_mb = max(batch_max_gpu_peak_mem_mb, repeat_peak_total_mb)
                            batch_had_valid_mem = True

                        median_ms = float(np.median(latencies))
                        mean_ms   = float(np.mean(latencies))
                        std_ms    = float(np.std(latencies))
                        p99_ms    = float(np.percentile(latencies, 99))
                        ips       = b / (median_ms / 1000.0)

                        repeat_rows.append({
                            "timestamp_utc": datetime.utcnow().isoformat(),
                            "device": device_name,
                            "sweep_id": s,
                            "repeat_id": r,
                            "model": model_name,
                            "batch_size": b,
                            "threads": t,
                            "iters": TIMED_ITERS,
                            "warmup_iters": WARMUP_ITERS,
                            "median_ms": round(median_ms, 3),
                            "mean_ms": round(mean_ms, 3),
                            "std_latency_ms": round(std_ms, 3),
                            "p99_latency_ms": round(p99_ms, 3),
                            "throughput_ips": round(ips, 3),
                            "repeat_peak_total_mb": repeat_peak_total_mb
                        })

                    if skip_batch:
                        continue

                    final_gpu_peak_mem_mb = round(batch_max_gpu_peak_mem_mb, 1) if batch_had_valid_mem else ""

                    for row in repeat_rows:
                        with open(CSV_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([
                                row["timestamp_utc"],
                                row["device"],
                                row["sweep_id"],
                                row["repeat_id"],
                                row["model"],
                                row["batch_size"],
                                row["threads"],
                                row["iters"],
                                row["warmup_iters"],
                                row["median_ms"],
                                row["mean_ms"],
                                row["std_latency_ms"],
                                row["p99_latency_ms"],
                                row["throughput_ips"],
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
                                final_gpu_peak_mem_mb
                            ])

                        mem_str = f" | PeakUsedMem(maxR): {final_gpu_peak_mem_mb:>6} MB" if use_cuda else ""

                        print(
                            f"{row['model']:<10} | S:{row['sweep_id']:02d} | R:{row['repeat_id']} | B:{row['batch_size']:<3} | "
                            f"Med: {row['median_ms']:10.2f} ms | "
                            f"P99: {row['p99_latency_ms']:10.2f} ms | "
                            f"{row['throughput_ips']:8.2f} ips{mem_str}"
                        )

print("-" * 140)
print("Benchmark Complete.")
