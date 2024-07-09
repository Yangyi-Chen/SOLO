import math
import lz4.frame
import pathlib
import zstandard as zstd
import argparse
import threading
import ujson as json
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--input-files', type=str, default="data/raw/slimpajama/SlimPajama-627B/train/**/*.jsonl.zst")
parser.add_argument('--output-filepath', type=str, help='Output file', default='data/processed/slimpajama/slimpajama.shard_{shard_id:03d}.jsonl.lz4')
parser.add_argument('--meta-to-remove', type=str, default="RedPajamaCommonCrawl,RedPajamaC4")
parser.add_argument('--n-workers', type=int, default=50)
parser.add_argument('--n-output-shards', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

META_TO_REMOVE = set(args.meta_to_remove.split(","))
pathlib.Path(args.output_filepath).parent.mkdir(parents=True, exist_ok=True)

# assign jsonl files to shards
jsonl_files = glob(args.input_files)
n_files_per_shard = math.ceil(len(jsonl_files) / args.n_output_shards)
print(f"Processing {len(jsonl_files)} jsonl files")
print(f"Writing to {args.output_filepath} with {args.n_output_shards} shards ({n_files_per_shard} files per shard)")
pbar = tqdm(total=len(jsonl_files))

def jsonl_reader(filepath: str) -> Iterable[Dict]:
    with open(filepath, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            previous = b''
            while True:
                chunk = reader.read(2**20)
                if not chunk:
                    break
                previous += chunk
                eol = previous.find(b'\n')
                while eol != -1:
                    line = previous[:eol]
                    previous = previous[eol+1:]
                    yield json.loads(line)
                    eol = previous.find(b'\n')
        if previous:
            yield json.loads(previous)

def producer(filepaths: List[str], queue: mp.Queue):
    """
    {
    "text": ...,
    "meta": {"redpajama_set_name": "RedPajamaCommonCrawl" | "RedPajamaC4" | "RedPajamaGithub" | "RedPajamaBook" | "RedPajamaArXiv" | "RedPajamaWikipedia" | "RedPajamaStackExchange"},
    }
    """

    print(f"Process {mp.current_process().name} started for processing {len(filepaths)} files.")
    for filepath in filepaths:
        for instance in jsonl_reader(filepath):
            meta = instance["meta"]
            if meta["redpajama_set_name"] in META_TO_REMOVE:
                queue.put("REMOVED_DUE_TO_META")
                continue
            # SAVE OUTPUT
            queue.put({
                "role": "assistant",
                "content": instance["text"],
            })
        queue.put("END_OF_FILE")
    print(f"Process {mp.current_process().name} finished processing {len(filepaths)} files.")

stats_counter = defaultdict(int)
stats_counter["n_instances"] = 0
for shard_id in range(args.n_output_shards):
    start = shard_id * n_files_per_shard
    end = min((shard_id + 1) * n_files_per_shard, len(jsonl_files))
    cur_shard_filepaths = jsonl_files[start:end]
    pbar.set_description(f"Processing Shard {shard_id}: {start}-{end}")

    # PRODUCER
    # manually start multiple processes and communicate via a queue
    queue = mp.Queue(maxsize=args.n_workers * 1024)
    processes = []
    for i in range(args.n_workers):
        cur_worker_filepaths = cur_shard_filepaths[i::args.n_workers]
        process = mp.Process(target=producer, args=(cur_worker_filepaths, queue))
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
            elif content == "REMOVED_DUE_TO_META":
                stats_counter["n_instances_removed_due_to_meta"] += 1
                continue

            assert isinstance(content, dict)
            fout.write(json.dumps(content) + "\n")
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
