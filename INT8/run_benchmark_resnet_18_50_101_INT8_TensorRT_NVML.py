#!/usr/bin/env python3
"""
TensorRT ResNet benchmark (GPU) with batch sweeps + P99 + NVML total GPU memory.

What this script does:
1) Exports ResNet models from PyTorch to ONNX (once).
2) Builds TensorRT engines:
   - FP16 engine (no calibration)
   - INT8 engine (with Entropy calibrator + calibration cache)
3) Runs inference at batch sizes [1,2,4,8,16,32,64,128,256,384,512],
   measures median/mean/std/P99 + throughput,
   and appends to benchmark_results_final.csv.

Memory behavior:
- Uses NVML total GPU memory used, not allocation delta.
- For each repeat, tracks the peak total used memory observed.
- For each batch, computes the max of all repeat peaks.
- Writes that same batch-level peak memory value for every repeat row
  so charts stay consistent across precisions.
"""

import os
import csv
import platform
import subprocess
from datetime import datetime

import numpy as np

import torch
import torchvision
from torchvision import models

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

# ----------------------------
# Config
# ----------------------------
CSV_FILE = "benchmark_results_final.csv"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
WARMUP_ITERS = 20
TIMED_ITERS = 100
SWEEPS_S = 10
REPEATS_R = 3

# Engine build config
MAX_BATCH = max(BATCH_SIZES)
INPUT_C, INPUT_H, INPUT_W = 3, 224, 224

# Choose which TensorRT precisions to run
RUN_FP16 = False
RUN_INT8 = True

# Where to store ONNX + TRT engines + calibration caches
ARTIFACT_DIR = "trt_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ----------------------------
# NVML total GPU used memory
# ----------------------------
import pynvml

pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def try_get_total_used_mem_mb_via_nvml():
    """
    Return total GPU memory currently used on device 0, in MB.
    This is total system usage reported by NVML, not allocation delta.
    """
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        return round(info.used / (1024 * 1024), 1)
    except Exception as e:
        print(f"[WARN] NVML read failed: {e}")
        return None


# ----------------------------
# Sysinfo (same style)
# ----------------------------
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

# Keep SAME schema as earlier scripts
header = [
    "timestamp_utc", "device", "sweep_id", "repeat_id", "model", "batch_size",
    "threads", "iters", "warmup_iters", "median_ms", "mean_ms", "std_latency_ms", "p99_latency_ms",
    "throughput_ips", "hostname", "os", "kernel", "arch", "cpu_model",
    "cpu_sockets", "cpu_cores_per_socket", "cpu_threads_per_core",
    "cpu_logical_cpus", "mem_total_mb", "python_version", "torch_version",
    "cuda_available", "torchvision_version", "gpu_peak_mem_mb"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

# ----------------------------
# ONNX export
# ----------------------------
def export_resnet_to_onnx(model_name: str, onnx_path: str):
    if os.path.exists(onnx_path):
        return

    if model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().cuda()
    elif model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().cuda()
    elif model_name == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).eval().cuda()
    else:
        raise ValueError("unknown model")

    x = torch.randn(1, INPUT_C, INPUT_H, INPUT_W, device="cuda")

    torch.onnx.export(
        m,
        x,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"[INFO] Exported {model_name} -> {onnx_path}")


# ----------------------------
# INT8 calibrator
# ----------------------------
class RandomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Simple calibrator that feeds random data.
    For better INT8 accuracy, replace random data with real samples.
    """
    def __init__(self, cache_file: str, batch_size: int = 32, n_batches: int = 50):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batch_count = 0

        self.input_shape = (self.batch_size, INPUT_C, INPUT_H, INPUT_W)
        nbytes = np.prod(self.input_shape) * np.float32().nbytes
        self.d_input = cuda.mem_alloc(int(nbytes))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_count >= self.n_batches:
            return None

        host_data = np.random.randn(*self.input_shape).astype(np.float32)
        cuda.memcpy_htod(self.d_input, host_data)

        self.batch_count += 1
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ----------------------------
# Build TensorRT engine
# ----------------------------
def build_engine(onnx_path: str, engine_path: str, precision: str):
    """
    precision: "fp16" or "int8"
    """
    if os.path.exists(engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("Failed to deserialize existing engine")
            return engine

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        cache_file = engine_path + ".calib"
        config.int8_calibrator = RandomEntropyCalibrator(
            cache_file=cache_file,
            batch_size=32,
            n_batches=50
        )
    else:
        raise ValueError("precision must be 'fp16' or 'int8'")

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile.set_shape(
        input_name,
        min=(1, INPUT_C, INPUT_H, INPUT_W),
        opt=(min(32, MAX_BATCH), INPUT_C, INPUT_H, INPUT_W),
        max=(MAX_BATCH, INPUT_C, INPUT_H, INPUT_W),
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build engine: {engine_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    if engine is None:
        raise RuntimeError("Failed to deserialize CUDA engine")

    print(f"[INFO] Built TensorRT engine: {engine_path} ({precision})")
    return engine


# ----------------------------
# Allocate buffers + run
# ----------------------------
def allocate_io(engine: trt.ICudaEngine, batch_size: int):
    context = engine.create_execution_context()

    input_name = None
    output_name = None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError("Failed to find input/output tensor names")

    context.set_input_shape(input_name, (batch_size, INPUT_C, INPUT_H, INPUT_W))

    in_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    out_shape = tuple(context.get_tensor_shape(output_name))
    out_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    nbytes_in = int(np.prod((batch_size, INPUT_C, INPUT_H, INPUT_W)) * np.dtype(in_dtype).itemsize)
    nbytes_out = int(np.prod(out_shape) * np.dtype(out_dtype).itemsize)

    d_input = cuda.mem_alloc(nbytes_in)
    d_output = cuda.mem_alloc(nbytes_out)

    h_output = np.empty(out_shape, dtype=out_dtype)

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    stream = cuda.Stream()

    return context, input_name, output_name, d_input, d_output, h_output, stream


def timed_inference_ms(context, input_name, d_input, stream, batch_size: int):
    h_input = np.random.randn(batch_size, INPUT_C, INPUT_H, INPUT_W).astype(np.float32)

    cuda.memcpy_htod_async(d_input, h_input, stream)

    start_evt = cuda.Event()
    end_evt = cuda.Event()

    start_evt.record(stream)
    context.execute_async_v3(stream_handle=stream.handle)
    end_evt.record(stream)

    end_evt.synchronize()
    return float(start_evt.time_till(end_evt))


# ----------------------------
# Main benchmark
# ----------------------------
def run_for_model(model_base: str, precision: str):
    device_name = "cuda"
    threads_logged = 1

    onnx_path = os.path.join(ARTIFACT_DIR, f"{model_base}.onnx")
    export_resnet_to_onnx(model_base, onnx_path)

    engine_path = os.path.join(ARTIFACT_DIR, f"{model_base}_{precision}_maxB{MAX_BATCH}.engine")
    engine = build_engine(onnx_path, engine_path, precision)

    for s in range(1, SWEEPS_S + 1):
        for b in BATCH_SIZES:
            context, input_name, output_name, d_input, d_output, h_output, stream = allocate_io(engine, b)

            # Store per-repeat benchmark rows first, then write them after batch max memory is known
            pending_rows = []
            batch_max_mem_mb = 0.0

            for r in range(1, REPEATS_R + 1):
                repeat_max_mem_mb = 0.0

                # Warmup
                for _ in range(WARMUP_ITERS):
                    _ = timed_inference_ms(context, input_name, d_input, stream, b)

                    mem_mb = try_get_total_used_mem_mb_via_nvml()
                    if mem_mb is not None and mem_mb > repeat_max_mem_mb:
                        repeat_max_mem_mb = mem_mb

                # One more sample after warmup sync point
                mem_mb = try_get_total_used_mem_mb_via_nvml()
                if mem_mb is not None and mem_mb > repeat_max_mem_mb:
                    repeat_max_mem_mb = mem_mb

                latencies = []

                for _ in range(TIMED_ITERS):
                    lat_ms = timed_inference_ms(context, input_name, d_input, stream, b)
                    latencies.append(lat_ms)

                    mem_mb = try_get_total_used_mem_mb_via_nvml()
                    if mem_mb is not None and mem_mb > repeat_max_mem_mb:
                        repeat_max_mem_mb = mem_mb

                if repeat_max_mem_mb > batch_max_mem_mb:
                    batch_max_mem_mb = repeat_max_mem_mb

                median_ms = float(np.median(latencies))
                mean_ms = float(np.mean(latencies))
                std_ms = float(np.std(latencies))
                p99_ms = float(np.percentile(latencies, 99))
                ips = b / (median_ms / 1000.0)

                pending_rows.append({
                    "repeat_id": r,
                    "median_ms": round(median_ms, 3),
                    "mean_ms": round(mean_ms, 3),
                    "std_ms": round(std_ms, 3),
                    "p99_ms": round(p99_ms, 3),
                    "ips": round(ips, 3),
                })

                repeat_mem_str = f" | RepeatPeakUsedMem: {repeat_max_mem_mb:>6.1f} MB" if repeat_max_mem_mb > 0 else ""
                print(
                    f"{model_base+'_'+precision:<14} | S:{s:02d} | R:{r} | B:{b:<3} | "
                    f"Med: {median_ms:10.2f} ms | P99: {p99_ms:10.2f} ms | {ips:8.2f} ips{repeat_mem_str}"
                )

            # Final batch-level peak total used memory across all repeats
            gpu_peak_mem_mb = round(batch_max_mem_mb, 1) if batch_max_mem_mb > 0 else ""

            # Write all repeat rows using the same batch-level memory value
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                for row in pending_rows:
                    writer.writerow([
                        datetime.utcnow().isoformat(),
                        device_name,
                        s,
                        row["repeat_id"],
                        f"{model_base}_{precision}",
                        b,
                        threads_logged,
                        TIMED_ITERS,
                        WARMUP_ITERS,
                        row["median_ms"],
                        row["mean_ms"],
                        row["std_ms"],
                        row["p99_ms"],
                        row["ips"],
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
                        gpu_peak_mem_mb
                    ])

            batch_mem_str = f"{gpu_peak_mem_mb:.1f} MB" if gpu_peak_mem_mb != "" else "N/A"
            print(
                f"[BATCH SUMMARY] {model_base+'_'+precision:<14} | S:{s:02d} | B:{b:<3} | "
                f"BatchPeakTotalUsedMem(maxR): {batch_mem_str}"
            )

            del context, d_input, d_output


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. TensorRT benchmarking requires a GPU.")

    try:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version: {torch.version.cuda}")
    except Exception:
        pass

    print(f"Logging to: {CSV_FILE} (Updated in real-time)")
    print("-" * 140)

    model_list = ["resnet101", "resnet50", "resnet18"]

    for m in model_list:
        if RUN_FP16:
            run_for_model(m, "fp16")
        if RUN_INT8:
            run_for_model(m, "int8")

    print("-" * 140)
    print("Benchmark Complete.")


if __name__ == "__main__":
    main()
