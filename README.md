# SOLO: A Single Transformer for Scalable Vision-Language Modeling

## Setup

### Clone Repo

```bash
git clone https://github.com/Yangyi-Chen/SOLO
git submodule update --init --recursive
```

### Setup Environment for Data Processing

```bash
conda env create -f environment.yml
conda activate solo
```

OR simply

```bash
pip install -r requirements.txt
```

## Inference SOLO with huggingface transformers

Then you can run [`scripts/notebook/hf_model_infer.ipynb`](scripts/notebook/hf_model_infer.ipynb) to perform inference on the model.


## Training

Please refer to [PRETRAIN_GUIDE.md](PRETRAIN_GUIDE.md) for more details about how to perform pre-training.

## Citations

```bibtex
TODO
```
