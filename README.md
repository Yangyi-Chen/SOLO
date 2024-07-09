<h1 align="center"> SOLO: A Single Transformer for Scalable Vision-Language Modeling </h1>

<p align="center">
<a href="https://arxiv.org/abs/TODO">📃 Paper</a>
•
<a href="https://huggingface.co/xingyaoww/SOLO-7B" >🤗 Model (SOLO-7B)</a>
</p>

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
