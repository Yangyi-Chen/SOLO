import io
import torch
import base64
import requests
import torchvision.transforms as transforms
from math import ceil
from PIL import Image
import matplotlib.pyplot as plt

PATCH_SIZE = 32
MAX_RESOLUTION = 1024 # 32 * 32

def get_resize_output_image_size(
    image_size,
) -> tuple:
    #
    return 1024, 1024
    l1, l2 = image_size # 540, 32
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


def preprocess_image(
    image_tensor: torch.Tensor,
    patch_size=PATCH_SIZE
) -> torch.Tensor:
    # Reshape the image to get the patches
    # shape changes: (C=3, H, W) 
    # -> (C, N_H_PATCHES, W, PATCH_H)
    # -> (C, N_H_PATCHES, N_W_PATCHES, PATCH_H, PATCH_W)
    patches = image_tensor.unfold(1, patch_size, patch_size)\
        .unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous() # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches


def get_transform(height, width):
    preprocess_transform = transforms.Compose([
            transforms.Resize((height, width)),             
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                std=[0.229, 0.224, 0.225])   # standard deviation for pre-trained models on ImageNet
        ])
    return preprocess_transform

def get_reverse_transform():
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage()
    ])
    return reverse_transform

def load_image_to_base64(image_path: str) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def download_image_to_base64(image_url: str) -> str:
    response = requests.get(image_url)
    img_base64 = base64.b64encode(response.content).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def load_image_bytes_to_base64(image_bytes: bytes) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def load_image_bytes_to_PILImage(image_bytes: bytes) -> Image:
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

def load_base64_to_PILImage(base64_string: str) -> Image:
    # convert data:image/jpeg;base64, to jpeg
    base64_string = base64_string.split(",")[1]
    decoded_string = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded_string)).convert('RGB')

def convert_image_base64_to_patches(base64_image: str) -> torch.Tensor:
    img_pil = load_base64_to_PILImage(base64_image)
    # resize the image to the nearest multiple of 32
    width, height = img_pil.size
    new_width, new_height = get_resize_output_image_size((width, height))
    img_tensor = get_transform(new_height, new_width)(img_pil) # 3， height, width 
    # transform the process img to seq_length, 64*64*3
    img_patches = preprocess_image(img_tensor) # seq_length, 64*64*3
    return img_patches


def visualize_patches(img_patches, figsize=(6, 6)):
    assert len(img_patches.shape) == 5, "Input should be a 5D tensor"
    n_width_patches, n_height_patches = img_patches.shape[1], img_patches.shape[0]
    width, height = n_width_patches * PATCH_SIZE, n_height_patches * PATCH_SIZE

    # Calculate the total width and height with black lines included
    total_width = width + (n_width_patches - 1)  # add space for black lines between patches
    total_height = height + (n_height_patches - 1)  # add space for black lines between patches

    # Create an empty image to place the patches
    full_image = Image.new('RGB', (total_width, total_height))

    for row_id in range(img_patches.shape[0]):
        for col_id in range(img_patches.shape[1]):
            patch = img_patches[row_id, col_id]
            patch = get_reverse_transform()(patch)
            # Calculate top left position of where to paste the patch
            top = row_id * (PATCH_SIZE + 1)  # include space for a black line
            left = col_id * (PATCH_SIZE + 1)  # include space for a black line
            full_image.paste(patch, (left, top))

    # Visualize the full image
    plt.figure(figsize=figsize)
    plt.imshow(full_image)
    plt.axis('off')  # Hide axes ticks
    plt.show()
    
    