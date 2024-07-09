import os
import json
import json
import gzip
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm

from src.data.utils import load_image_to_base64

tqdm.pandas()

parser = argparse.ArgumentParser(description='Process LLAVA-CC3M data')
parser.add_argument('--input-filepath', type=str, help='Input file', default='data/raw/LLaVA-CC3M-Pretrain-595K/chat.json')
parser.add_argument('--image-dir', type=str, help='Input file', default='data/raw/LLaVA-CC3M-Pretrain-595K/images')
parser.add_argument('--output-filepath', type=str, help='Output file', default='data/processed/llava_cc3m.jsonl.gz')
args = parser.parse_args()

# Load data
with open(args.input_filepath, 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# df['image'] = df['image'].progress_apply(lambda x: load_image_to_base64(os.path.join(args.image_dir, x)))

# Example of a conversation
# "messages": [
#     {
#     "role": "user",
#     "content": [
#         {
#         "type": "text",
#         "text": "What’s in this image?"
#         },
#         {
#         "type": "image_url",
#         "image_url": {
#             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#         }
#         }
#     ]
#     },
# ]
def normalize_conversation(conv, cur_img_filename: str):
    normalized_conv = []
    cur_img_base64 = load_image_to_base64(os.path.join(args.image_dir, cur_img_filename))
    for turn in conv:
        cur_role = {
            "human": "user",
            "gpt": "assistant"
        }[turn['from']]

        if cur_role == "user":
            text_chunks = turn['value'].split("<image>")
            assert len(text_chunks) == 2, "Expecting only one image per turn for LLaVA-CC3M"
            # insert image between text chunks
            cur_msg = {
                "role": cur_role,
                "content": [
                    {
                        "type": "text",
                        "text": text_chunks[0]
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": cur_img_base64}
                    },
                ]
            }
            if text_chunks[1]:  # if there's text after the image
                cur_msg["content"].append({
                    "type": "text",
                    "text": text_chunks[1]
                })
        else:
            cur_msg = {
                "role": cur_role,
                "content": [
                    {
                        "type": "text",
                        "text": turn['value']
                    }
                ]
            }
        normalized_conv.append(cur_msg)

    return normalized_conv

# df['conversations'] = df.progress_apply(lambda x: normalize_conversation(x['conversations'], x['image']), axis=1)
# df[["id", "conversations"]].to_json("data/processed/llava_cc3m.json.gz", orient="records", lines=True, compression="gzip")

pathlib.Path(args.output_filepath).parent.mkdir(parents=True, exist_ok=True)

# streaming write to gz
with gzip.open(args.output_filepath, "wt") as f:
    for i, row in tqdm(df.iterrows(), total=len(df)):
        conv = normalize_conversation(row['conversations'], row['image'])
        f.write(json.dumps({
            "id": row['id'],
            "conversations": conv
        }) + "\n")
