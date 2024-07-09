import os
import math
import random
import pathlib
import lz4.frame
import tarfile
import argparse
import threading
import ujson as json
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict

from src.data.utils import load_image_bytes_to_base64

tqdm.pandas()

parser = argparse.ArgumentParser(description='Process ImageNet-21k data')
parser.add_argument('--input-filepath', type=str, help='Input file', default='/shared/nas/data/m1/yangyic3/MultimodalAgent/data/imagenet-w21-wds/mapping.txt')
parser.add_argument('--image-dir', type=str, help='Input file', default='/shared/nas/data/m1/yangyic3/MultimodalAgent/data/imagenet-w21-wds/')
parser.add_argument('--output-filepath', type=str, help='Output file', default='data/processed/imagenet21k/imagenet21k.shard_{shard_id:03d}.jsonl.lz4')
parser.add_argument('--n-workers', type=int, default=50)
parser.add_argument('--n-output-shards', type=int, default=32)
parser.add_argument('--before-ratio', type=float, default=1.0) # default to always insert before text
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Set seed
random.seed(args.seed)


# Load data
id_to_name = {}
with open(args.input_filepath, 'r') as f:
    for line in f:
        _id, name = line.strip().split()
        id_to_name[_id] = name

pathlib.Path(args.output_filepath).parent.mkdir(parents=True, exist_ok=True)

tar_filepaths = sorted(glob(os.path.join(args.image_dir, "*.tar")))
n_files_per_shard = math.ceil(len(tar_filepaths) / args.n_output_shards)
print(f"Processing {len(tar_filepaths)} tar files")
print(f"Writing to {args.output_filepath} with {args.n_output_shards} shards ({n_files_per_shard} files per shard)")


pbar = tqdm(total=len(tar_filepaths))

def get_data_instances(tar_filepaths, queue) -> Tuple[List[Dict], Dict]:
    print(f"Process {mp.current_process().name} started for processing {len(tar_filepaths)} files.")
    for tar_filepath in tar_filepaths:
        with tarfile.open(tar_filepath) as tar:
            stat_counter = defaultdict(int)
            # get a list of files
            files = tar.getnames()
            jpgs = [f for f in files if f.endswith(".jpg")]
            keys = [f.split(".")[0] for f in jpgs]
            for key in keys:
                with tar.extractfile(f"{key}.jpg") as f:
                    # load into bytes
                    image_bytes = f.read()
                
                class_id = key.split("_")[0]
                text = id_to_name[class_id].replace("_", " ").replace("-", " ")
                
                # YIELD OUTPUT
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": load_image_bytes_to_base64(image_bytes)}
                }

                stat_counter["n_images_inserted"] += 1
                # flatten content: list of list of content -> list of content
                content = [image_content]
                content.append({"type": "text", "text": text})

                queue.put(({
                    # since we are doing pre-training, we just set
                    # the role to assistant for all instances
                    # (this is required for training pipeline)
                    "role": "assistant",
                    "content": content
                }, stat_counter))

        queue.put("END_OF_FILE")
    print(f"Process {mp.current_process().name} finished.")

stats_counter = defaultdict(int)
stats_counter["n_instances"] = 0
for shard_id in range(args.n_output_shards):
    start = shard_id * n_files_per_shard
    end = min((shard_id + 1) * n_files_per_shard, len(tar_filepaths))
    cur_shard_filepaths = tar_filepaths[start:end]
    pbar.set_description(f"Processing Shard {shard_id}: {start}-{end}")

    # PRODUCER
    # manually start multiple processes and communicate via a queue
    queue = mp.Queue(maxsize=args.n_workers * 1024)
    processes = []
    for i in range(args.n_workers):
        cur_worker_tar_filepaths = cur_shard_filepaths[i::args.n_workers]
        process = mp.Process(target=get_data_instances, args=(cur_worker_tar_filepaths, queue))
        process.start()
        processes.append(process)

    # CONSUMER
    def writer(queue, fout):
        print("Starting writer thread")
        while True:
            content = queue.get()
            if content is None:
                # None is the signal that the process is done
                break
            elif content == "END_OF_FILE":
                pbar.update(1)
                continue

            instance, cur_stats_counter = content
            fout.write(json.dumps(instance) + "\n")
            # add stats
            for k, v in cur_stats_counter.items():
                stats_counter[k] += v
            pbar.set_postfix(stats_counter)
            stats_counter["n_instances"] += 1

    # with gzip.open(args.output_filepath.format(shard_id=shard_id), "wt") as fout:
    with lz4.frame.open(args.output_filepath.format(shard_id=shard_id), "wt") as fout:
        # start a writer thread
        writer_thread = threading.Thread(target=writer, args=(queue, fout))
        writer_thread.start()

        # wait for all processes to finish
        for process in processes:
            process.join()
        
        # wait for the writer thread to finish
        queue.put(None)
        writer_thread.join()

pbar.close()

with open(args.output_filepath.replace(".shard_{shard_id:03d}.jsonl.gz", ".stats.json"), "w") as f:
    f.write(json.dumps(stats_counter, indent=4))
