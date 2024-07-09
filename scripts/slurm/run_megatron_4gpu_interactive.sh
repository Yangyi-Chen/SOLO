#!/bin/bash
srun \
    -A bcdy-delta-gpu \
    --time=00:30:00 \
    --nodes=1 \
    --ntasks-per-node=64 \
    --tasks=1 \
    --cpus-per-task=64 \
    --partition=gpuA40x4 \
    --gpus=4 \
    --mem=240g \
    --pty scripts/slurm/run_megatron_interactive.sh
