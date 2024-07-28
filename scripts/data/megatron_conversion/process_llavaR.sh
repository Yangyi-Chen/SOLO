#!/bin/bash

set -e
TOKENIZER_PATH=data/models/raw/MultimodalMistral-7B-megatron/tokenizer.model


DATA_INPUT="/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/llavaR.jsonl.gz"
OUTPUT_PREFX=/shared/rsaas/xingyao6/projects/Multimodal-Mistral/data/processed/megatron_format/mmistral_llavaR_pack32k/data
mkdir -p $(dirname $OUTPUT_PREFX)

python Megatron-LLM/tools/preprocess_multimodal_data.py \
    --input $DATA_INPUT \
    --output_prefix $OUTPUT_PREFX \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file $TOKENIZER_PATH \
    --no_mp \
    --no_new_tokens \
    --do_pretrain \
    --do_packing \
    --max_seq_length 32768 \
    --log_interval 100
