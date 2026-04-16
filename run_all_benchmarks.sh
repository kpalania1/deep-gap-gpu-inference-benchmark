#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

BASE_DIR="$(pwd)"

run_job() {
    local workdir="$1"
    local script_name="$2"
    local label="$3"

    echo "=================================================="
    echo "Starting $label at $(date)"
    echo "Workdir : $workdir"
    echo "Script  : $script_name"
    echo "GPU     : $CUDA_DEVICE"
    echo "=================================================="

    cd "$workdir"

    env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
        python3 -u "$script_name" \
        > "output_${label}.log" 2>&1

    echo "Finished $label at $(date)"
    echo
}

run_job "$BASE_DIR/FP32" "run_benchmark_resnet_18_50_101_FP32_NVML.py" "FP32"

echo "Cooling GPU for 10 minutes..."
sleep 600

run_job "$BASE_DIR/FP16" "run_benchmark_resnet_18_50_101_FP16_NVML.py" "FP16"

echo "Cooling GPU for 10 minutes..."
sleep 600

run_job "$BASE_DIR/INT8-TensorRT" "run_benchmark_resnet_18_50_101_INT8_TensorRT_NVML.py" "INT8"

echo "All benchmark runs completed at $(date)"
