#!/bin/bash

HF_FORMAT_DIR=data/models/raw_hf/Mistral-7B-v0.1
MEGATRON_FORMAT_DIR=data/models/raw/Mistral-7b-megatron

python Megatron-LLM/weights_conversion/hf_to_megatron.py mistral \
    --size=7 \
	--out=$MEGATRON_FORMAT_DIR \
    --model-path=$HF_FORMAT_DIR \
