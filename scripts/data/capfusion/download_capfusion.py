import pathlib
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="data/raw/capfusion-120m")
args = parser.parse_args()

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

dataset = load_dataset("BAAI/CapsFusion-120M")
df = dataset["train"].to_pandas()
df = df[["image_url", "capsfusion"]].rename(
    columns={"image_url": "url", "capsfusion": "caption"}
)
df.to_parquet(f"{args.output_dir}/url_and_caption.parquet")
