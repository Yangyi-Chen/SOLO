#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

MAPPING_PARQUET=data/raw/mmc4/data/images_no_face_v3.selected.parquet
INPUT_DOCS_GLOB="data/raw/mmc4/data/docs_no_face_v3/*.jsonl"
# INPUT_IMAGES_DIR=data/raw/mmc4/data/images_no_face_v3
# SSD disk for faster processing
INPUT_IMAGES_DIR=/scratch/xingyao6/Multimodal-Mistral/data/raw/mmc4/data/images_no_face_v3

N_OUTPUT_SHARDS=32
N_WORKERS=64

# mkdir -p data/processed/mmc4
# OUTPUT_FILEPATH="data/processed/mmc4/mmc4.shard_{shard_id:03d}.jsonl.gz"
mkdir -p /scratch/xingyao6/Multimodal-Mistral/data/processed/mmc4
OUTPUT_FILEPATH="/scratch/xingyao6/Multimodal-Mistral/data/processed/mmc4/mmc4.shard_{shard_id:03d}.jsonl.gz"

BEFORE_RATIO=1.0  # make all images come before text

python3 scripts/data/mmc4/process_mmc4.py \
    --input-mapping-parquet $MAPPING_PARQUET \
    --input-docs-glob "$INPUT_DOCS_GLOB" \
    --input-images-dir $INPUT_IMAGES_DIR \
    --n-output-shards $N_OUTPUT_SHARDS \
    --n-workers $N_WORKERS \
    --output-filepath $OUTPUT_FILEPATH \
    --before-ratio $BEFORE_RATIO \
    --remove-instances-missing-images
