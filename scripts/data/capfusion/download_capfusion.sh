#!/bin/bash

# Download dataset to data/raw/capfusion-120m/url_and_caption.parquet
python3 scripts/data/capfusion/download_capfusion.py --output_dir data/raw/capfusion-120m

URL_LIST=data/raw/capfusion-120m/url_and_caption.parquet
RAW_IMG_DIR=data/raw/capfusion-120m/images_and_caption
mkdir -p $RAW_IMG_DIR
N_PARALLEL=8

# Faster way using img2dataset
echo "Make sure you installed img2dataset (pip install img2dataset)"

# use img2dataset to download
img2dataset \
    --input_format=parquet \
    --url_list=$URL_LIST \
    --output_folder=$RAW_IMG_DIR \
    --processes_count=$N_PARALLEL \
    --url_col=url \
    --caption_col=caption \
    --image_size=1024 \
    --resize_mode=keep_ratio_largest \
    --resize_only_if_bigger=True \
    --output_format=webdataset \
    --enable_wandb=True 2>&1 | tee -a capfusion_img_download.log

