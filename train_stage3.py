"""Stage 3: fine-tune an embedding model on the PawSaDaan training cats.

ArcFace over the 53 training cats, P x K batches, only the last
`--trainable-depth` backbone stages unfrozen; best epoch by PawSaDaan validation
mAP (see training.py). `--init` starts from a Stage 2 checkpoint instead of the
pretrained backbone (the ArcFace class weights always start fresh).

    python train_stage3.py --backbone dinov2_vits14 --backbone-lr 3e-5
    python train_stage3.py --init runs/<stage2 run>/best.pt
    python train_stage3.py --backbone resnet50 --backbone-lr 3e-4

Validation cats are never trained on; the test split is never touched here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data import stage3_rows
from training import DEFAULT_DEPTH, add_training_arguments, fit, run_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_training_arguments(parser, epochs=30)
    parser.add_argument("--supplemental", action="store_true",
                        help="also train on supplemental_for_primary photos")
    parser.add_argument("--init", type=Path, help="Stage 2 checkpoint to start from")
    args = parser.parse_args()
    if args.init and not args.init.is_file():
        parser.error(f"--init not found: {args.init}")
    depth = args.trainable_depth or DEFAULT_DEPTH[args.backbone]
    suffix = ("_supp" if args.supplemental else "") + (
        f"_from-{args.init.resolve().parent.name}" if args.init else "")
    rows = stage3_rows("train", include_supplemental=args.supplemental)
    fit(args, "stage3", rows, run_name("stage3", args, depth, suffix), depth, init=args.init)


if __name__ == "__main__":
    main()
