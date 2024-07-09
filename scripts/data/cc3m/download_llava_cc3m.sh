#!/bin/bash

echo "Please make sure git-lfs is installed!"
mkdir data/raw
pushd data/raw
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
cd LLaVA-CC3M-Pretrain-595K
unzip images.zip -d images
