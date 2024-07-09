import os
import json
import json
import gzip
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm

from src.data.utils import load_image_to_base64, load_base64_to_PILImage

tqdm.pandas()

parser = argparse.ArgumentParser(description='Process LLAVA-CC3M data')
parser.add_argument('--input-filepath', type=str, help='Input file', default='data/raw/imagenet/category.jsonl')
parser.add_argument('--image-dir', type=str, help='Input file', default='/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/imagenet/train/')
parser.add_argument('--output-filepath', type=str, help='Output file', default='data/processed/imagenet_train.jsonl.gz')
args = parser.parse_args()

# Load data
id_to_name = {}
with open(args.input_filepath, 'r') as f:
    for line in f:
        data = json.loads(line)
        id_to_name[data['id']] = data['name']

def create_conversation(text, cur_img_filename: str):
    cur_img_base64 = load_image_to_base64(os.path.join(args.image_dir, cur_img_filename))

    # test if the image is valid - will raise an error if not
    load_base64_to_PILImage(cur_img_base64)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": cur_img_base64}
                },
            ]
        },
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


pathlib.Path(args.output_filepath).parent.mkdir(parents=True, exist_ok=True)

# streaming write to gz
with gzip.open(args.output_filepath, "wt") as f:

    class_pbar = tqdm(os.listdir(args.image_dir), desc="Processing Classes")
    n_total = 0
    n_failed = 0

    for folder_name in class_pbar:
        try:
            name = id_to_name[folder_name]
        except:
            print(f"Failed to process {folder_name}")
            continue

        file_pbar = tqdm(os.listdir(os.path.join(args.image_dir, folder_name)), desc=folder_name)
        for image_filename in file_pbar:
            image_filename = f"{folder_name}/{image_filename}"
            
            n_total += 1
            try:
                text = " ".join(name.split("_")).lower()
                conv = create_conversation(text, image_filename)
                f.write(json.dumps({
                    "id": image_filename,
                    "conversations": conv
                }) + "\n")
            except Exception as e:
                print(f"Failed to process {image_filename}: {e}")
                n_failed += 1
            
            file_pbar.set_postfix({"n_failed": n_failed, "n_total": n_total})
print(f"Failed to process {n_failed} out of {n_total} images")
