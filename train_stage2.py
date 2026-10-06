"""Stage 2: pre-train an embedding model on the Hello Street Cat training cats.

ArcFace over the 38 HSC training cats (1,885 photos), same recipe as Stage 3
(see training.py). The epoch is selected on PawSaDaan validation mAP, because
the goal is a better starting point for PawSaDaan, not better HSC scores.
Continue with: python train_stage3.py --init runs/<this run>/best.pt

    python train_stage2.py                      # DINOv2 ViT-S/14, backbone LR 3e-5
"""
from __future__ import annotations

import argparse

from data import stage2_rows
from training import DEFAULT_DEPTH, add_training_arguments, fit, run_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_training_arguments(parser, epochs=20)
    args = parser.parse_args()
    depth = args.trainable_depth or DEFAULT_DEPTH[args.backbone]
    fit(args, "stage2", stage2_rows("train"), run_name("stage2", args, depth), depth)


if __name__ == "__main__":
    main()
