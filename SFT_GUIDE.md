# Instruction Fine-Tuning Guide

## Data Downloading
The data will be uploaded to <a href="https://huggingface.co/datasets/YangyiYY/VLM-SFT" >🤗 VLM-SFT</a> soon.
```bash
# assume you have installed git-lfs. If not, please run conda install git-lfs.
git clone https://huggingface.co/datasets/YangyiYY/VLM-SFT
```



## Model Training
You should first specify the `train_data`, `img_dir`, `proj_dir`, `checkpoint` in the `config/SFT.yml` file:
1. `train_data`: the path to the training data `all_data.jsonl` (downloaded in the previous step).
2. `img_dir`: the path to the image directory. The images are downloaded in the previous step.
3. `proj_dir`: the name for wandb logger. You can set it to your own project name.
4. `checkpoint`: the path to the pre-trained model. See [PRETRAIN_GUIDE.md](PRETRAIN_GUIDE.md) for the pre-training instructions. 

Then run:
```bash
scripts/sft/run.sh
```

The output will be saved to `data/ckpts/SFT/`.
