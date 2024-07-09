#!/bin/bash
mkdir -p data/raw/slimpajama
cd data/raw/slimpajama
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
