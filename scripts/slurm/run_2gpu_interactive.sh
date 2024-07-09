#!/bin/bash
srun \
    -A bcbf-delta-gpu \
    --time=00:30:00 \
    --nodes=1 \
    --ntasks-per-node=16 \
    --tasks=2 \
    --cpus-per-task=16 \
    --partition=gpuA40x4 \
    --gpus=2 \
    --mem=128g \
    --pty /bin/bash
