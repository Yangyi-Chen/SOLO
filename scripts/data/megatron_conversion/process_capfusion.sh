#!/bin/bash

set -e
TOKENIZER_PATH=data/models/raw/MultimodalMistral-7B-megatron/tokenizer.model

# ========================================
# Capfusion
shard_ids=(
    000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 
    016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031
    032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047
    048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063
    064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079
    080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095
    096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111
    112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
)

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
    
    DATA_INPUT="data/processed/capfusion/capfusion.shard_${shard_id}.jsonl.lz4"
    OUTPUT_PREFX=data/processed/megatron_format/mmistral_capfusion_shard_${shard_id}_pack32k/data

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
        OUTPUT_LOG=logs/data/megatron_format_mmistral_capfusion_shard_${shard_id}_pack32k.log
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

