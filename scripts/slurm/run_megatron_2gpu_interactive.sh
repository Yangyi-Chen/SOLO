#!/bin/bash
srun \
    -A bcdy-delta-gpu \
    --time=00:30:00 \
    --nodes=1 \
    --ntasks-per-node=32 \
    --tasks=1 \
    --cpus-per-task=32 \
    --partition=gpuA40x4 \
    --gpus=2 \
    --mem=120g \
    --pty scripts/slurm/run_megatron_interactive.sh
