#!/bin/bash

set -e
TOKENIZER_PATH=data/models/raw/MultimodalMistral-7B-megatron/tokenizer.model

# ========================================
# Capfusion
shard_ids=(000)

RUN_IN_BACKGROUND=1
N_PARALLEL=3

# Create an array to store background process IDs
declare -a PIDS=()

# Add this line at the beginning of the script to import the signal module
trap 'kill_background_processes' INT

function kill_background_processes() {
    echo -e '\nReceived Ctrl+C. Killing background processes...'

    # Loop through the stored background process IDs and kill them
    for pid in "${PIDS[@]}"; do
        if ps -p $pid >/dev/null; then
            echo "Killing process $pid"
            kill $pid
        fi
    done

    # Exit the script
    exit 1
}

function remove_from_array {
    local -n arr=$1
    local value=$2

    # Find the index of the element to be removed
    local index=0
    for i in "${arr[@]}"; do
        if [ "$i" = "$value" ]; then
            break
        fi
        index=$((index + 1))
    done

    # Remove the element at the found index
    unset 'arr[$index]'

    # Reindex the array
    arr=("${arr[@]}")
}

for shard_id in ${shard_ids[@]}; do
    # original
    
    DATA_INPUT="/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/processed/detailed_captions/detailed_captions.shard_${shard_id}.jsonl.lz4"
    OUTPUT_PREFX=data/processed/megatron_format/mmistral_detailed_captions_shard_${shard_id}_pack32k/data

    if [ -f "${OUTPUT_PREFX}_DONE" ]; then
        echo "Skipping $DATA_INPUT since it is already completed"
        continue
    fi

    # if directory exists, remove it
    if [ -d "$(dirname $OUTPUT_PREFX)" ]; then
        echo "Removing existing directory $(dirname $OUTPUT_PREFX)"
        rm -r $(dirname $OUTPUT_PREFX)
    fi
    mkdir -p $(dirname $OUTPUT_PREFX)
    

    if (($RUN_IN_BACKGROUND)); then
        OUTPUT_LOG=logs/data/megatron_format_mmistral_detailed_captions_shard_${shard_id}_pack32k.log
        python -u Megatron-LLM/tools/preprocess_multimodal_data.py \
            --input $DATA_INPUT \
            --output_prefix $OUTPUT_PREFX \
            --tokenizer_type SentencePieceTokenizer \
            --vocab_file $TOKENIZER_PATH \
            --no_mp \
            --no_new_tokens \
            --do_pretrain \
            --do_packing \
            --max_seq_length 32768 \
            --log_interval 100 > $OUTPUT_LOG 2>&1 &

        cur_pid=$!
        echo -e "\n** Started process $cur_pid (run in background). To track progress, run:"
        echo -e "  tail -f $OUTPUT_LOG"

        # Store the background process ID in the array
        PIDS+=("$cur_pid")
        # 2>&1 | tee -a $output_dir/output.txt

        # Control the number of parallel processes by waiting for some to finish
        # Adjust the value after -le to set the desired number of parallel processes
        while ((${#PIDS[@]} >= N_PARALLEL)); do
            for pid in "${PIDS[@]}"; do
                if ! ps -p "$pid" >/dev/null; then
                    # Remove the finished process from the array
                    echo "Process $pid finished. Remaining processes: ${PIDS[@]}"
                    remove_from_array PIDS "$pid"
                fi
            done
            # Sleep for a short time before checking again
            sleep 1
        done
    fi

    python -u Megatron-LLM/tools/preprocess_multimodal_data.py \
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
        #  > $OUTPUT_LOG 2>&1 &

done

