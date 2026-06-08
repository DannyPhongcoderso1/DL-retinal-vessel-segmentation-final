"""Export DRIVE test tensors to PNG images for app testing.

Example:
    python src/export_test_images.py \
        --data dataset/drive_test_dataset.pt \
        --output outputs/app_test_images_full
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test dataset PNG images.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/drive_test_dataset.pt"),
        help="Path to the .pt test dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/app_test_images_full"),
        help="Directory where PNG images and masks will be saved.",
    )
    return parser.parse_args()


def tensor_to_image_array(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float()
    if image.dim() == 4:
        image = image.squeeze(0)

    if image.dim() == 3 and image.size(0) == 3:
        # The exported DRIVE test dataset is ImageNet-normalized.
        image = image * IMAGENET_STD + IMAGENET_MEAN
        array = image.clamp(0, 1).permute(1, 2, 0).numpy()
    elif image.dim() == 3 and image.size(-1) == 3:
        array = image.numpy()
        if array.max() > 1.5:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)
    elif image.dim() == 2:
        array = image.numpy()
        if array.max() > 1.5:
            array = array / 255.0
        array = np.stack([np.clip(array, 0.0, 1.0)] * 3, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    return (array * 255.0).round().astype(np.uint8)


def tensor_to_mask_array(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu().float()
    if mask.dim() == 4:
        mask = mask.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if mask.dim() != 2:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    return ((mask.numpy() > 0.5).astype(np.uint8) * 255)


def unpack_sample(sample: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(sample, dict):
        image = sample.get("image") or sample.get("images") or sample.get("x")
        mask = None
        for key in ("mask", "manual", "manual_1", "manual_2", "target", "label", "y"):
            if key in sample:
                mask = sample[key]
                break
        if image is None or mask is None:
            raise ValueError(f"Cannot find image/mask keys in sample: {sample.keys()}")
        return image, mask

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]

    raise ValueError(f"Unsupported sample type: {type(sample)}")


def iter_samples(data: Any) -> list[Any]:
    if isinstance(data, dict) and "images" in data and ("masks" in data or "labels" in data):
        masks = data["masks"] if "masks" in data else data["labels"]
        return list(zip(data["images"], masks))
    if isinstance(data, torch.utils.data.TensorDataset):
        return list(data)
    if isinstance(data, (list, tuple)):
        return list(data)

    raise ValueError(f"Unsupported dataset object: {type(data)}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.data, map_location="cpu")
    samples = iter_samples(data)

    for index, sample in enumerate(samples):
        image, mask = unpack_sample(sample)
        image_path = args.output / f"drive_test_{index:02d}_image.png"
        mask_path = args.output / f"drive_test_{index:02d}_mask.png"
        Image.fromarray(tensor_to_image_array(image)).save(image_path)
        Image.fromarray(tensor_to_mask_array(mask)).save(mask_path)

    print(f"Exported {len(samples)} samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
