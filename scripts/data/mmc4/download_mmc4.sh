#!/bin/bash

cwd=$(pwd)

mkdir -p data/raw
pushd data/raw
git clone https://github.com/allenai/mmc4.git
cd mmc4
# at data/raw/mmc4

# Download MMC4 Files
mkdir -p data/docs_no_face_v3
chmod +x scripts/fewer_faces_corev3.sh
./scripts/fewer_faces_corev3.sh data/docs_no_face_v3

# Unzip Files
find data/docs_no_face_v3 -name "*.zip" | xargs -P 8 unzip

# # Slower way to download image
# for shard_file in $shard_files; do
#     output_image_dir="data/images_no_face_v3/$(basename $shard_file .jsonl)"
#     mkdir -p $output_image_dir
#     echo "Downloading images for $shard_file to $output_image_dir"
#     python scripts/download_images.py \
#         --input_jsonl $shard_file \
#         --output_image_dir $output_image_dir \
#         --num_process 48
# done

cd $cwd

# Download Images
DATA_DIR=data/raw/mmc4/data/docs_no_face_v3
URL_DIR=data/raw/mmc4/data/image_urls_no_face_v3
RAW_IMG_DIR=data/raw/mmc4/data/images_no_face_v3
N_PARALLEL=32
shard_files=$(ls $DATA_DIR/*.jsonl)

# Faster way using img2dataset
echo "Make sure you installed img2dataset (pip install img2dataset)"

# Process shard files in parallel
mkdir -p $URL_DIR
ls $DATA_DIR/*.jsonl | xargs -P $N_PARALLEL -I {} python3 -u scripts/data/mmc4/get_urls.py --input_jsonl {} --output_dir $URL_DIR

# use img2dataset to download
mkdir -p $RAW_IMG_DIR
img2dataset \
    --input_format=parquet \
    --url_list=$URL_DIR \
    --output_folder=$RAW_IMG_DIR \
    --processes_count=$N_PARALLEL \
    --image_size=1024 \
    --resize_mode=keep_ratio_largest \
    --resize_only_if_bigger=True \
    --output_format=webdataset \
    --enable_wandb=True 2>&1 | tee -a mmc4_img_download.log

echo "Download complete! Please use scripts/notebook/view_mmc4_downloaded.ipynb to filter out bad images for future processing."
