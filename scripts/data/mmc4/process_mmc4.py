import math
import gzip
import random
import tarfile
import argparse
import itertools
import ujson as json
import pandas as pd
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from src.data.utils import load_image_bytes_to_base64

parser = argparse.ArgumentParser()
parser.add_argument('--input-mapping-parquet', type=str, default="data/raw/mmc4/data/images_no_face_v3.selected.parquet")
parser.add_argument('--input-docs-glob', type=str, default="data/raw/mmc4/data/docs_no_face_v3/*.jsonl")
parser.add_argument('--input-images-dir', type=str, default="data/raw/mmc4/data/images_no_face_v3")
parser.add_argument('--output-filepath', type=str, help='Output file', default='data/processed/mmc4.shard_{shard_id:03d}.jsonl.gz')
parser.add_argument('--n-workers', type=int, default=4)
parser.add_argument('--chunk-size', type=int, default=32)
parser.add_argument('--n-output-shards', type=int, default=128)
parser.add_argument('--before-ratio', type=float, default=1.0) # default to always insert before text
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--remove-instances-missing-images', action="store_true")
args = parser.parse_args()

# Set seed
random.seed(args.seed)

# def initializer(args):
#     global mapping

#     print(f"Initalizing process {mp.current_process().name}.")

#     # Load the mapping
#     mapping = pd.read_parquet(args.input_mapping_parquet)
#     mapping = mapping[["img2dataset_shard_id", "key", "url"]]
#     mapping["tar_filepath"] = mapping["img2dataset_shard_id"].apply(lambda x: f"{args.input_images_dir}/{x}.tar")
#     mapping = mapping[['url', 'tar_filepath', 'key']]
#     # mapping url to (tar_filepath, key)
#     mapping = mapping.set_index("url")

#     print(f"Process {mp.current_process().name} initialized.")

# Load the mapping
mapping = pd.read_parquet(args.input_mapping_parquet)
mapping = mapping[["img2dataset_shard_id", "key", "url"]]
mapping["tar_filepath"] = mapping["img2dataset_shard_id"].apply(lambda x: f"{args.input_images_dir}/{x:05d}.tar")
mapping = mapping[['url', 'tar_filepath', 'key']]
# mapping url to (tar_filepath, key)
mapping = mapping.set_index("url")

def load_image(tar_filepath, key) -> bytes:
    with tarfile.open(tar_filepath) as tar:
        with tar.extractfile(f"{key}.jpg") as f:
            # load into bytes
            return f.read()
        

def convert_data_instance(jsonl_row: str) -> Optional[Tuple[List[Dict], Dict]]:
    """    
    # Example
    {'image_info': [{'face_detections': None,
                 'image_name': 'b9040a0dbb22.jpg',
                 'matched_sim': 0.27694183588027954,
                 'matched_text_index': 2,
                 'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.90.jpg'},
                {'face_detections': None,
                 'image_name': 'db1c21bc8474.jpg',
                 'matched_sim': 0.3234919607639313,
                 'matched_text_index': 1,
                 'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.91.jpg'}],
    'similarity_matrix': [[0.24363446235656738,
                            0.31758785247802734,
                            0.27694183588027954],
                        [0.2233106791973114,
                            0.3234919607639313,
                            0.26118797063827515]],
    'text_list': ['When you lock the door using the lock tab on the driver’s '
                'door, all of the other doors and tailgate lock at the same '
                'time.',
                'Press the master door lock switch in as shown to lock or '
                'unlock all doors and the tailgate.',
                'When you lock/unlock the driver’s door and tailgate using the '
                'master lock switch, all the other doors lock/ unlock at the '
                'same time.'],
    'url': 'http://www.hfitinfo.com/hofi-48.html',
    'could_have_url_duplicate': 0 }
    """
    stat_counter = defaultdict(int)
    text_list = jsonl_row["text_list"]
    images_insert_before_text = [ [] for _ in range(len(text_list)) ]
    images_insert_after_text = [ [] for _ in range(len(text_list)) ]

    for image_info in jsonl_row["image_info"]:
        # randomly decide whether to prepend or append the image to the corresponding text
        insert_before = random.random() < args.before_ratio
        try:
            mapped_to = mapping.loc[image_info["raw_url"]]
            tar_filepath = mapped_to["tar_filepath"]
            key = mapped_to["key"]
        except KeyError:
            if args.remove_instances_missing_images:
                stat_counter["instance_skipped_due_to_missing_image"] += 1
                return None  # skip this instance
            else:
                stat_counter["n_missing_images"] += 1
                continue # skip this image
        
        # Process image
        image_bytes = load_image(tar_filepath, key)
        image_base64 = load_image_bytes_to_base64(image_bytes)
        image_content = {
            "type": "image_url",
            "image_url": {"url": image_base64}
        }

        stat_counter["n_images_inserted"] += 1

        if insert_before:
            stat_counter["n_images_inserted_before_text"] += 1
            images_insert_before_text[image_info["matched_text_index"]].append(image_content)
        else:
            stat_counter["n_images_inserted_after_text"] += 1
            images_insert_after_text[image_info["matched_text_index"]].append(image_content)

    # flatten content: list of list of content -> list of content
    content = []
    for i, text in enumerate(text_list):
        content.extend(images_insert_before_text[i])
        content.append({"type": "text", "text": text})
        content.extend(images_insert_after_text[i])

    return [
        {
            # since we are doing pre-training, we just set
            # the role to assistant for all instances
            # (this is required for training pipeline)
            "role": "assistant",
            "content": content
        }
    ], stat_counter


# Load the docs
docs_filepaths = glob(args.input_docs_glob)
assert len(docs_filepaths) == 23085
n_files_per_shard = math.ceil(len(docs_filepaths) / args.n_output_shards)

pbar = tqdm(total=len(docs_filepaths))

def jsonl_generator_fn(filepath):
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)
    pbar.update(1)

stats_counter = defaultdict(int)
for shard_id in range(args.n_output_shards):
    start = shard_id * n_files_per_shard
    end = min((shard_id + 1) * n_files_per_shard, len(docs_filepaths))
    pbar.set_description(f"Processing Shard {shard_id}: {start}-{end}")
    
    # Build generator for parallel processing
    jsonl_generator = itertools.chain.from_iterable(
        map(jsonl_generator_fn, docs_filepaths[start:end])
    )

    # Process the data
    # , initializer=initializer, initargs=(args,)
    with mp.Pool(args.n_workers) as pool, \
        gzip.open(args.output_filepath.format(shard_id=shard_id), "wt") as fout:
            instances_generator = pool.imap(convert_data_instance, jsonl_generator, args.chunk_size)
            # instances_generator = map(convert_data_instance, jsonl_generator)
            for content in instances_generator:
                if content is not None:
                    (instances, cur_stats_counter) = content
                    fout.write(json.dumps(instances) + "\n")
                    # add stats
                    for k, v in cur_stats_counter.items():
                        stats_counter[k] += v
                    pbar.set_postfix(stats_counter)

pbar.close()

with open(args.output_filepath.replace(".shard_{shard_id:03d}.jsonl.gz", ".stats.json"), "w") as f:
    f.write(json.dumps(stats_counter, indent=4))
