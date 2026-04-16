#!/usr/bin/env bash
nohup nvidia-smi \
  --query-gpu=timestamp,index,name,temperature.gpu,power.draw,pstate,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used \
  --format=csv,nounits \
  -l 1 > gpu_telemetry_full_run.csv 2>&1 &
echo $! > telemetry.pid
echo "Telemetry started with PID $(cat telemetry.pid)"
