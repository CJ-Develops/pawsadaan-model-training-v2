"""Shared paths and constants for the v2 training workspace."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Photos are read in place from the Source Folder; they are never copied here.
# Manifest image_path values are relative to SOURCE_ROOT.
SOURCE_ROOT = PROJECT_ROOT.parent / "PawSaDaan Source Folder"
DATASET_ROOT = SOURCE_ROOT / "PawSaDaan AI Training Collated v2"
UPSTREAM_MANIFEST_DIR = DATASET_ROOT / "03_TRAINING_PREPARATION" / "manifests"

# Frozen copy of the upstream manifests; see freeze_manifests.py.
MANIFEST_DIR = PROJECT_ROOT / "manifests"
FROZEN_RECORD = MANIFEST_DIR / "FROZEN.json"
MANIFEST_FILES = (
    "hellostreetcat_stage2.csv",
    "pawsadaan_stage3.csv",
    "pawsadaan_session_collection_needed.csv",
    "split_summary.json",
)

RUNS_DIR = PROJECT_ROOT / "runs"        # checkpoints and logs (gitignored)
RESULTS_DIR = PROJECT_ROOT / "results"  # metric reports (committed)

SEED = 42

# Matches the preprocessing contract of pawsadaan-ai-service. Both candidate
# backbones (ImageNet ResNet50, DINOv2 ViT-S/14) take 224px ImageNet-normalized input.
PREPROCESS = dict(
    size=224,
    resize=232,
    interpolation="bilinear",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    exif_transpose=True,
)
