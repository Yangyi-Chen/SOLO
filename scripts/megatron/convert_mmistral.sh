#!/bin/bash

# original model
# HF_FORMAT_DIR=data/models/raw_hf/MultimodalMistral-7B
# MEGATRON_FORMAT_DIR=data/models/raw/MultimodalMistral-7B-megatron

# cyy distilled model
# HF_FORMAT_DIR=data/models/raw_hf/MultimodalMistral-7B-distilled-proj
# MEGATRON_FORMAT_DIR=data/models/raw/MultimodalMistral-7B-dproj-megatron

# HF_FORMAT_DIR=data/ckpts/MultimodalMistral-7B-dproj-T2P1-train_imagenet21k_img70pct-no_sep_loss-32k-lr5e-5-warmup200-bs128-seq32768/hf/model_iter_1525
# MEGATRON_FORMAT_DIR=data/models/raw/MultimodalMistral-7B-dproj-megatron-imagenet21k-1525

# HF_FORMAT_DIR=data/ckpts/MultimodalMistral-7B-dproj-T2P1-train_1sSlimpajama+96sCapfusion+stage2-no_sep_loss-32k-lr5e-5-warmup200-bs128-seq32768/hf/model_iter_150_chat_added
# MEGATRON_FORMAT_DIR=data/models/raw/MultimodalMistral-7B-dproj-megatron-capfusion-150

# HF_FORMAT_DIR=data/ckpts1/MultimodalMistral-7B-dproj-T2P1-train_1sSlimpajama_capfusion32s_cc3m_detailed_cap+websight_llavaR_dvqa_ocrvqa_figureqa+stage2-no_sep_loss-32k-lr5e-5-warmup200-bs128-seq32768/hf/model_iter_3481
# MEGATRON_FORMAT_DIR=data/models/raw/MultimodalMistral-7B-dproj-megatron-caption+ocr-stage2-3481

HF_FORMAT_DIR=/shared/nas2/xingyao6/projects/Multimodal-Mistral/data/ckpts/MultimodalMistral-7B-dproj-T2P1-train_1sSlimpajama+cc3m_full_detailed_cap+stage3-no_sep_loss-32k-lr5e-5-warmup200-bs128-seq32768/hf/model_iter_300
MEGATRON_FORMAT_DIR=/shared/nas2/xingyao6/projects/Multimodal-Mistral/data/models/raw/MultimodalMistral-7B-dproj-megatron-caption+ocr-stage3-300

export PYTHONPATH=`pwd`:$PYTHONPATH
# TRUE_VOCAB_SIZE=33029
# 32000 + 5 (reserved vision token) + 1024 (location tokens)

python Megatron-LLM/weights_conversion/hf_to_megatron.py multimodal_mistral \
    --size=7 \
	--out=$MEGATRON_FORMAT_DIR \
    --model-path=$HF_FORMAT_DIR
