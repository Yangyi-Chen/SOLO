# Pretrain Guide

## Data Processing

1. All the raw data should be processed to a unified [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat) we supported. See [this](scripts/data/cc3m/process_llava_cc3m.py) for example.
Image will be reshaped and formatted as `base64` string for next step. Raw data will typically stored as `jsonl` or `jsonl.gz` (a compressed version), where each line is (there might be some variance - but the next step should be pretty agnostic to this):

```
{
    "messages": [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "IMAGE IN BASE64"}
                },
                {
                    "type": "text",
                    "text": "CAPTION HERE"
                }
            ]
        }
    ]
}
```

2. Then, you follow scripts under [scripts/data/megatron_conversion](scripts/data/megatron_conversion) to convert the raw dataset into megatron compatible dataset. For example: [scripts/data/megatron_conversion/process_full_cc3m.sh](scripts/data/megatron_conversion/process_full_cc3m.sh)

3. Then you should be done with the dataset proprocessing.


## Model Conversion

Megatron uses a special model format that requires multiple step of conversion.

1. Clone the raw mistral model to `data/models/raw_hf`

```bash
mkdir -p data/models/raw_hf
cd data/models/raw_hf
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
```

2. Convert Mistral (LLM) to MMistral (multi-modal version we built, aka SOLO), It will create new tokenizers and resize embedding for multimodal purpose:
```bash
python3 scripts/model/create_solo.py
```


3. Convert MMistral (HF) to megatron version, output to `data/models/raw/MultimodalMistral-7B-megatron`
```bash
# See commented lines in the script and change to the right source when necessary
scripts/megatron/convert_mmistral.sh
```

4. Shard megatron version to support tensor-parallel (TP) training:
```bash
# See commented lines in the script and change to the right source when necessary
# scripts/megatron/shard_model.sh [TP] [PP]
# generally running on 8*A100/H100 80G with 7B model, you should do TP=2:
scripts/megatron/shard_model.sh 2
```

5. You will see the output model at `data/models/sharded/MultimodalMistral-7B-megatron-tp2-pp1`. Now you can start training!

## Model Training

1. You need to generate a dataset mixture you want to use for training, as well as calculate the number of training data point. 
You can run [this script](scripts/notebook/analyze_tokens.ipynb) to generate those.

2. Copy paste the data mixture you got from that script into a bash script, set the training data (and other hyperparameter) accordingly. See [`scripts/megatron/train/curriculum/cont_pretrain_5e-5_imagenet21k_stage1.sh`](scripts/megatron/train/curriculum/cont_pretrain_5e-5_imagenet21k_stage1.sh) for example.

3. Enter an interactive docker environment for training: `scripts/docker/run_megatron_interactive.sh`

4. Run the command (inside container) and the training should start!

```bash
scripts/megatron/train/curriculum/cont_pretrain_5e-5_imagenet21k_stage1.sh
```

## Convert Megatron format back to huggingface for evaluation

The checkpoints will be saved to a following folder structure:

```
data
  - ckpts
    - MultimodalMistral-7B-dproj-SOMESUFFIX
      - iter_0000050
      - iter_0000100
      ...
```

You can run the following **inside docker** if you want to convert checkpoint for iteration `100`:

```bash
scripts/megatron/convert_sharded_to_hf.sh data/ckpts/MultimodalMistral-7B-dproj-SOMESUFFIX 100
```

The output will be saved to `data/ckpts/MultimodalMistral-7B-dproj-SOMESUFFIX/hf/model_iter_100`.
