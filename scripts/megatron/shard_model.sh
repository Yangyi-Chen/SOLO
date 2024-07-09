#!/bin/bash

TP=$1
PP=$2
MODEL_NAME=MultimodalMistral-7B-megatron

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is not set"
    exit 1
fi
if [ -z "$TP" ]; then
    echo "Defaulting to TP=1"
    TP=1
fi
if [ -z "$PP" ]; then
    echo "Defaulting to PP=1"
    PP=1
fi

echo "Sharding model with TP=$TP and PP=$PP"
MODEL_TYPE=multimodal_mistral
MODEL_DIR=data/models/raw/${MODEL_NAME}

VOCAB_PATH=data/models/raw/${MODEL_NAME}/tokenizer.model
TRUE_VOCAB_SIZE=33029  # 32000 + 5 (reserved vision token) + 1024 (location tokens)

mkdir -p data/models/sharded
OUTPUT_DIR=data/models/sharded/${MODEL_NAME}-tp${TP}-pp${PP}

python Megatron-LLM/tools/checkpoint_util.py \
	--target_tensor_parallel_size $TP \
	--target_pipeline_parallel_size $PP \
    --true_vocab_size=$TRUE_VOCAB_SIZE \
	--load_dir $MODEL_DIR \
	--save_dir $OUTPUT_DIR \
	--model_type $MODEL_TYPE \
	--bf16
