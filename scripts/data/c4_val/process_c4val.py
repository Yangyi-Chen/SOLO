import gzip
import pathlib
import argparse
import ujson as json
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--output-filepath', type=str, help='Output file', default='data/processed/c4_val.jsonl.gz')
args = parser.parse_args()

dataset = load_dataset("c4", "en", split="validation")
df = dataset.to_pandas()
print(f"Loaded {len(df)} examples.")

with gzip.open(args.output_filepath, "wt") as f:
    for i, row in df.iterrows():
        text = row["text"]
        message = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
        f.write(json.dumps(message) + "\n")
