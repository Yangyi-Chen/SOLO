#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPU_PER_NODE=8

TP=2
PP=1
_MP=$((TP*PP))
DP=$((N_GPU_PER_NODE/_MP))
echo "TP=$TP"
echo "PP=$PP"
echo "DP=$DP"

SEQ_LENGTH=32768
# MODEL_NAME=MultimodalMistral-7B
MODEL_NAME=MultimodalMistral-7B-dproj
MODEL_TYPE=multimodal_mistral
# ========================================
# LEARNING

# Following Olmo-7b
# https://arxiv.org/pdf/2402.00838.pdf
# 3e-4
LR=5e-5
MIN_LR=5e-6
WD=0.1
LR_DECAY_STYLE=cosine
# ========================================
# ========================================
# DATA

# TODO: CHANGE THIS FOR DIFFERENT DATASET
EXP_DATA_NOTE=train_imagenet21k_img70pct-no_sep_loss-32k

TRAIN_DATA_PREFIX="0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_000_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_001_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_002_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_003_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_004_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_005_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_006_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_007_pack32k/data 0.0218 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_008_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_009_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_010_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_011_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_012_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_013_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_014_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_015_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_016_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_017_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_018_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_019_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_020_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_021_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_022_pack32k/data 0.0218 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_023_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_024_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_025_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_026_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_027_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_028_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_029_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_030_pack32k/data 0.0219 /scratch/xingyao6/Multimodal-Mistral/data/processed/megatron_format/mmistral_imagenet21k_shard_031_pack32k/data 0.3000 data/processed/megatron_format/mmistral_slimpajama_shard_002_pack32k/data"
N_TRAIN_DATA=195122

# SCRATCH_PREFIX="/scratch/xingyao6/Multimodal-Mistral"
# VALID_DATA_PREFIX=${SCRATCH_PREFIX}/data/processed/megatron_format/mmistral_c4_val_25pct_128x11_pack32k/data
# N_VALID_DATA=1408
# TEST_DATA_PREFIX=${SCRATCH_PREFIX}/data/processed/megatron_format/mmistral_llava_cc3m_128x11_pack32k/data
# N_TEST_DATA=1408

# ========================================
# ========================================
# SETTINGS
EVAL_INTERVAL=10000000 # manually eval
SAVE_INTERVAL=50
N_WARMUP_STEPS=200

# each instance is 32k tokens, each batch is typically 4M tokens -> 4M / 32k = 128
# GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
GRAD_ACCUM=$((128/$DP))
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE*DP*GRAD_ACCUM))
N_EPOCHS=1
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "N_EPOCHS=$N_EPOCHS"

# -----------------------------------------
# -- automatically set the following variables --
N_BATCHES_PER_EPOCH=$((($N_TRAIN_DATA/$GLOBAL_BATCH_SIZE)+1))
TRAIN_ITERATIONS=$(($N_BATCHES_PER_EPOCH*$N_EPOCHS))
# VALID_ITERATIONS=$(($N_VALID_DATA/$GLOBAL_BATCH_SIZE))
# TEST_ITERATIONS=$(($N_TEST_DATA/$GLOBAL_BATCH_SIZE))
echo "N_BATCHES_PER_EPOCH=$N_BATCHES_PER_EPOCH"
echo "TRAIN_PREFIX=$TRAIN_DATA_PREFIX"
echo "TRAIN_ITERATIONS=$TRAIN_ITERATIONS"
# echo "VAL_PREFIX=$VALID_DATA_PREFIX"
# echo "VALID_ITERATIONS=$VALID_ITERATIONS"
# echo "TEST_PREFIX=$TEST_DATA_PREFIX"
# echo "TEST_ITERATIONS=$TEST_ITERATIONS"
# -----------------------------------------
# ========================================
# ========================================
# MODEL

# Automatically set the following variables
if [ -z "$EXP_ID" ]; then
    EXP_ID="${MODEL_NAME}-T${TP}P${PP}-${EXP_DATA_NOTE}-lr${LR}-warmup${N_WARMUP_STEPS}-bs${GLOBAL_BATCH_SIZE}-seq${SEQ_LENGTH}"
fi
if [ ! -z "$EXTRA_EXP_NOTE" ]; then
    EXP_ID="${EXP_ID}-N${EXTRA_EXP_NOTE}"
fi
MODEL_CKPT_DIR=data/ckpts/${EXP_ID}

# first check LOAD_MODEL_CKPTS, if it is not empty, then we load the model from the ckpt dir
# otherwise, if there is a model in the ckpt dir (check whether $MODEL_CKPT_DIR/latest_checkpointed_iteration.txt exists)
# then we load the model from the ckpt dir

if [ ! -z "$LOAD_MODEL_CKPTS" ]; then
    echo "Loading model from specified ckpt dir $LOAD_MODEL_CKPTS"
    MODEL_WEIGHT_DIR=$LOAD_MODEL_CKPTS
elif [ -f "$MODEL_CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "Found existing ckpt dir $MODEL_CKPT_DIR, loading from there"
    MODEL_WEIGHT_DIR=$MODEL_CKPT_DIR
else
    echo "No existing ckpt dir $MODEL_CKPT_DIR, loading from pretrained model"
    MODEL_WEIGHT_DIR=data/models/sharded/${MODEL_NAME}-megatron-tp${TP}-pp${PP}
fi

TOKENIZER_PATH=data/models/raw/${MODEL_NAME}-megatron/tokenizer.model
echo "EXP_ID=$EXP_ID"
echo "MODEL_CKPT_DIR=$MODEL_CKPT_DIR"
echo "MODEL_WEIGHT_DIR=$MODEL_WEIGHT_DIR"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
export WANDB_NAME=$EXP_ID
# export WANDB_MODE=disabled # debug

# ========================================
# ========================================
# COMPUTE EFFICIENCY

EFFICIENCY_ARGS="
    --recompute_granularity full
    --recompute_method uniform
    --recompute_num_layers 4
    --empty_unused_memory_level 1
    --use_distributed_optimizer
    --num_workers 8
    --use_flash_attn
"
# ========================================


# ========================================
# LAUNCH
# export CUDA_LAUNCH_BLOCKING=1 # only for DEBUG
FREQ_ARGS="--log_interval 1 --save_interval $SAVE_INTERVAL --eval_interval $EVAL_INTERVAL"
DATA_ARGS="
    --train_data_path $TRAIN_DATA_PREFIX
    --train_iters $TRAIN_ITERATIONS
    --packed_input
"
# --valid_data_path $VALID_DATA_PREFIX
# --valid_iters $VALID_ITERATIONS
# --test_data_path $TEST_DATA_PREFIX
# --test_iters $TEST_ITERATIONS
TRAIN_ARGS="
    --lr_decay_style $LR_DECAY_STYLE
    --lr_warmup_iters $N_WARMUP_STEPS
    --lr $LR --min_lr $MIN_LR --weight_decay $WD
    --clip_grad 1.0
"

DISTRIBUTED_ARGS="--nproc_per_node $N_GPU_PER_NODE --nnodes 1 --node_rank 0 --master_addr localhost --master_port 9999"

# dependency: build helper for blendable dataset
echo "Building Megatron-LLM/megatron/data for Blendable dataset"
pushd Megatron-LLM/megatron/data
make
popd
echo "Done building Megatron-LLM/megatron/data"

export CUDA_DEVICE_MAX_CONNECTIONS=1;
torchrun $DISTRIBUTED_ARGS Megatron-LLM/finetune.py \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP \
    --load $MODEL_WEIGHT_DIR \
    --save $MODEL_CKPT_DIR \
    --wandb_logger \
    --wandb_entity xingyaow \
    --wandb_project multimodal-mistral \
    --no_loss_beyond_token_id 32000 \
    --model_name $MODEL_TYPE \
    --tokenizer_type SentencePieceTokenizer \
    --vocab_file=$TOKENIZER_PATH \
    --no_new_tokens \
    --bf16 \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --data_type multimodal_instruction \
    --variable_seq_lengths \
    --no_bias_gelu_fusion \
    --no_bias_dropout_fusion \
    --hidden_dropout 0.0 \
    --attention_dropout 0.0 \
    --use_checkpoint_args \
    --exit_signal_handler \
    --save_on_exception \
    --override_opt_param_scheduler \
    $FREQ_ARGS $DATA_ARGS $TRAIN_ARGS $EFFICIENCY_ARGS
