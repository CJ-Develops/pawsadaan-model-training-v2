"""Manifest-driven row selection, image loading, and transforms.

Rows always come from the frozen manifests in manifests/, never from folder
listings, so every script sees exactly the same identity-disjoint splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
from functools import cache

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from config import (FROZEN_RECORD, MANIFEST_DIR, MANIFEST_FILES, PREPROCESS, RUNS_DIR,
                    SOURCE_ROOT)

SPLITS = ("train", "validation", "test")


@cache
def _verified_manifest_dir():
    if not FROZEN_RECORD.is_file():
        raise RuntimeError("No frozen manifests; run: python freeze_manifests.py")
    expected = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))["manifest_sha256"]
    for name in MANIFEST_FILES:
        actual = hashlib.sha256((MANIFEST_DIR / name).read_bytes()).hexdigest()
        if actual != expected[name]:
            raise RuntimeError(f"manifests/{name} changed after freezing.")
    return MANIFEST_DIR


def _read(name: str) -> list[dict]:
    with (_verified_manifest_dir() / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stage2_rows(split: str) -> list[dict]:
    """Hello Street Cat rows kept by the manifest (duplicates/low-count removed)."""
    if split not in SPLITS:
        raise ValueError(split)
    return [row for row in _read("hellostreetcat_stage2.csv")
            if row["split"] == split and row["include"] == "1"]


def stage3_rows(split: str, include_supplemental: bool = False) -> list[dict]:
    """PawSaDaan rows for one identity-disjoint split.

    Supplemental photos are excluded by default (include_initial_stage3=0).
    """
    if split not in SPLITS:
        raise ValueError(split)
    roles = {"primary", "supplemental_for_primary"} if include_supplemental else {"primary"}
    return [row for row in _read("pawsadaan_stage3.csv")
            if row["split"] == split and row["dataset_role"] in roles]


def distractor_rows() -> list[dict]:
    """Limited-identity PawSaDaan cats, never trained on; used as gallery decoys."""
    return [row for row in _read("pawsadaan_stage3.csv")
            if row["dataset_role"] == "limited_identity"]


def label_map(rows: list[dict]) -> dict[str, int]:
    return {identity: index for index, identity in
            enumerate(sorted({row["identity"] for row in rows}))}


def load_image(row: dict) -> Image.Image:
    with Image.open(SOURCE_ROOT / row["image_path"]) as image:
        if PREPROCESS["exif_transpose"]:
            image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(PREPROCESS["resize"],
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(PREPROCESS["size"]),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])


def train_transform() -> transforms.Compose:
    """Augmentation for uint8 CHW tensors from cached_images(rows, "train")."""
    return transforms.Compose([
        transforms.RandomResizedCrop(PREPROCESS["size"], scale=(0.6, 1.0), antialias=True,
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])


def normalize_uint8(images: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW batch -> float, ImageNet-normalized (same as ToTensor + Normalize)."""
    mean = torch.tensor(PREPROCESS["mean"], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(PREPROCESS["std"], device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255 - mean) / std


TRAIN_CACHE_SHORT_SIDE = 256


def cached_images(rows: list[dict], mode: str) -> torch.Tensor | list[torch.Tensor]:
    """Decode each photo once and keep it as uint8, cached under runs/cache/.

    mode "eval":  Resize 232 -> CenterCrop 224, stacked N x 3 x 224 x 224. Normalizing
                  it with normalize_uint8 gives exactly eval_transform()'s output.
    mode "train": shorter side resized to 256 (aspect kept), list of 3 x H x W, for
                  train_transform(). Phone photos are ~12 MP; decoding them every epoch
                  would dominate training time.
    """
    if mode == "eval":
        prepare = transforms.Compose([
            transforms.Resize(PREPROCESS["resize"],
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(PREPROCESS["size"]), transforms.PILToTensor()])
    elif mode == "train":
        prepare = transforms.Compose([
            transforms.Resize(TRAIN_CACHE_SHORT_SIDE,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.PILToTensor()])
    else:
        raise ValueError(mode)
    key = hashlib.sha256(json.dumps(
        [mode, TRAIN_CACHE_SHORT_SIDE, PREPROCESS, [row["sha256"] for row in rows]]).encode())
    path = RUNS_DIR / "cache" / f"{mode}_{key.hexdigest()[:20]}.pt"
    if path.is_file():
        return torch.load(path)
    images = [prepare(load_image(row)) for row in tqdm(rows, desc=f"cache {mode}", leave=False)]
    result = torch.stack(images) if mode == "eval" else images
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


class ManifestDataset(Dataset):
    """Yields (image_tensor, label_index, row_index) for a list of manifest rows."""

    def __init__(self, rows: list[dict], transform, labels: dict[str, int] | None = None):
        self.rows = rows
        self.transform = transform
        self.labels = labels or label_map(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return self.transform(load_image(row)), self.labels[row["identity"]], index
