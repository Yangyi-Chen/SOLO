import os
import json
import lz4.frame
import argparse
import multiprocessing as mp
import concurrent.futures
import threading

from glob import glob
from tqdm import tqdm

# ========
# Handle Multimodal Images
import io
import torch
import base64
import torchvision.transforms as transforms
from math import ceil
from PIL import Image

PATCH_SIZE = 32
MAX_RESOLUTION = 1024  # 32 * 32


def get_resize_output_image_size(
    image_size,
) -> tuple:
    l1, l2 = image_size  # the order of width/height should not matters
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    # set the nearest multiple of PATCH_SIZE for `long`
    requested_new_long = min(
        [
            ceil(long / PATCH_SIZE) * PATCH_SIZE,
            MAX_RESOLUTION,
        ]
    )

    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / PATCH_SIZE) * PATCH_SIZE
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)


def preprocess_image(image_tensor: torch.Tensor, patch_size=PATCH_SIZE) -> torch.Tensor:
    # Reshape the image to get the patches
    # shape changes: (C=3, H, W)
    # -> (C, N_H_PATCHES, W, PATCH_H)
    # -> (C, N_H_PATCHES, N_W_PATCHES, PATCH_H, PATCH_W)
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )

    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches


def get_transform(height, width):
    preprocess_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                std=[0.229, 0.224, 0.225],
            ),  # standard deviation for pre-trained models on ImageNet
        ]
    )
    return preprocess_transform


def get_reverse_transform():
    reverse_transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage(),
        ]
    )
    return reverse_transform


def load_image_to_base64(image_path: str) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def load_base64_to_PILImage(base64_string: str) -> Image:
    # convert data:image/jpeg;base64, to jpeg
    base64_string = base64_string.split(",")[1]
    decoded_string = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded_string)).convert("RGB")


# ============


def gather_image_stats_in_data(data: dict, filepath: str):

    stats = []
    # each line can be:
    if isinstance(data, list):
        # 1 conversations: [{"role": "human", "content": [{"type": "text", "text": "XXX"}]}]
        conversations = data
    elif isinstance(data, dict) and "conversations" in data:
        # 2 a dict with "id": {"id": "XXX", "conversations": [{"role": "human", "content": [{"type": "text", "text": "XXX"}]}]}
        assert isinstance(data, dict), "Data must be a list or a dictionary."
        # _id = data["id"]
        conversations = data["conversations"]
    elif isinstance(data, dict) and "content" in data and "role" in data:
        # 3 a dict with {"role": "human", "content": [{"type": "text", "text": "XXX"}]}
        conversations = [data]
    else:
        raise ValueError(f"Unknown data format: {data}")

    for turn in conversations:
        role = turn["role"]
        if not isinstance(turn["content"], list):
            assert isinstance(
                turn["content"], str
            ), "Content must be a string (text) if not a list."
            turn["content"] = [{"type": "text", "text": turn["content"]}]
        for item in turn["content"]:
            if item["type"] == "text":
                pass  #
            elif item["type"] == "image_url":
                # load image
                img_content = item["image_url"]["url"]
                assert (
                    "base64" in img_content
                ), "Only base64 image is currently supported."
                img_pil = load_base64_to_PILImage(img_content)
                width, height = img_pil.size
                new_width, new_height = get_resize_output_image_size((width, height))
                # img_tensor = get_transform(new_height, new_width)(img_pil)
                # cur_vision_patches = preprocess_image(img_tensor)

                # # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
                # n_h_patches, n_w_patches, c, patch_h, patch_w = cur_vision_patches.shape
                # n_patches = n_h_patches * n_w_patches
                # flatten the patches -> (N_PATCHES, C*PATCH_H*PATCH_W)
                # cur_vision_patches = cur_vision_patches.view(n_patches, -1)
                # n_images += 1
                stats.append(
                    {
                        "width": width,
                        "height": height,
                        "new_width": new_width,
                        "new_height": new_height,
                        "filepath": filepath,
                    }
                )
            else:
                raise ValueError(
                    f"Unknown content type (only 'text' and 'image_url' are supported): {item['type']}"
                )
    return stats


# ============

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_glob",
    type=str,
    default="/shared/nas2/xingyao6/projects/Multimodal-Mistral/data/processed/imagenet21k/imagenet21k.shard_*.jsonl.lz4",
)
parser.add_argument("--num_workers", type=int, default=32)
args = parser.parse_args()

data_files = sorted(glob(args.data_glob))
print(f"Found {len(data_files)} data files from {args.data_glob}.")
output_filepath = os.path.join(os.path.dirname(data_files[0]), "image_stats.jsonl")
print(f"Stats will be saved to {output_filepath}.")

manager = mp.Manager()
queue = manager.Queue()
files_pbar = tqdm(total=len(data_files), desc="Files")


def process_data_per_process(queue, data_file):
    print(f"Processing {data_file} in process {mp.current_process().pid}...")
    with lz4.frame.open(data_file, "rb") as f:
        for line in f:
            data = json.loads(line)
            stats = gather_image_stats_in_data(data, data_file.split("/")[-1])
            queue.put(stats)
    files_pbar.update(1)


instance_pbar = tqdm(desc="Instances")


def saver(queue):
    print("Starting saver thread...")
    with open(output_filepath, "w") as f:
        while True:
            stats = queue.get()
            if stats is None:
                break

            for stat in stats:
                f.write(json.dumps(stat) + "\n")
            f.flush()
            instance_pbar.update(1)
            if instance_pbar.n % 100 == 0:
                instance_pbar.set_postfix(
                    {k: v for k, v in stat.items() if k != "filepath"}
                )
    print("Saver thread finished.")


# run the saver in a separate thread
saver_thread = threading.Thread(target=saver, args=(queue,))
saver_thread.start()

# Create a pool of worker processes
with mp.Pool(args.num_workers) as pool:
    pool.starmap(
        process_data_per_process, [(queue, data_file) for data_file in data_files]
    )

# Send a sentinel to signal completion to the saver thread
queue.put(None)
saver_thread.join()  # Wait for the saver thread to finish
instance_pbar.close()
files_pbar.close()
print(f"Finished processing all data. Stats are saved to {output_filepath}.")
