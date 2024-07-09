#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH

# scripts/train/megatron/convert_sharded_to_hf.sh data/ckpts/Llama-2-7b-megatron-tp2-pp2 52
LOAD_DIR=$1 # data/ckpts/Llama-2-7b-megatron-tp2-pp2
LOAD_ITER=$2 # 300 to load the from iter_0000300 iteration
OUTPUT_DIR=$LOAD_DIR"/hf/model_iter_"$LOAD_ITER
echo "Converting sharded checkpoint $LOAD_DIR to huggingface $OUTPUT_DIR"

MODEL_TYPE=multimodal_mistral
MODEL_NAME=MultimodalMistral-7B-dproj
echo "MODEL_TYPE=$MODEL_TYPE"
echo "MODEL_NAME=$MODEL_NAME"

TOKENIZER_PATH=data/models/raw/${MODEL_NAME}-megatron/tokenizer.model
VOCAB_SIZE=33029  # 32000 + 5 (reserved vision token) + 1024 (location tokens)
echo "Use vocab file $TOKENIZER_PATH"

TMP_DIR=tmp/megatron-conversion
TMP_OUTPUT_DIR=$TMP_DIR/$LOAD_DIR
mkdir -p $TMP_OUTPUT_DIR

set -ex
# convert to unsharded checkpoint
python Megatron-LLM/tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir $LOAD_DIR \
	--load_iters $LOAD_ITER \
	--save_dir $TMP_OUTPUT_DIR \
	--model_type $MODEL_TYPE \
	--true_vocab_size $VOCAB_SIZE \
	--use_distributed_optimizer \
	--bf16

# convert to huggingface checkpoint
python Megatron-LLM/weights_conversion/megatron_to_hf.py \
	--model=$MODEL_TYPE \
	--input_dir=$TMP_OUTPUT_DIR \
	--output_dir=$OUTPUT_DIR \
	--vocab_file=$TOKENIZER_PATH \
	--no_new_tokens

# TODO: Add chat format
# python3 scripts/chat_interface/add_chat_format.py $OUTPUT_DIR

rm -rf $TMP_DIR
