"""Shared ArcFace training loop for train_stage2.py and train_stage3.py.

Batches are P cats x K photos from uint8 images cached in memory. After every
epoch the model is scored on PawSaDaan validation (+ 44 decoy cats); the epoch
with the best validation mAP is kept as runs/<name>/best.pt and its full report
(with bootstrap CIs) goes to results/pawsadaan_validation_<name>.json.
Runs in fp32 (fp16 gives NaN on the GTX 1660 SUPER; see CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import RUNS_DIR, SEED
from data import cached_images, label_map, train_transform
from evaluate import Evaluator, print_table, write_report
from models import ArcFace, EmbeddingModel, load_checkpoint, save_checkpoint

DEFAULT_DEPTH = {"dinov2_vits14": 4, "dinov2_vits14_reg4": 4, "resnet50": 1}


def add_training_arguments(parser: argparse.ArgumentParser, epochs: int) -> None:
    parser.add_argument("--backbone", default="dinov2_vits14", choices=sorted(DEFAULT_DEPTH))
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trainable-depth", type=int,
                        help="DINOv2: last N blocks (default 4); ResNet50: last N layers (default 1)")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--p", type=int, default=8, help="cats per batch")
    parser.add_argument("--k", type=int, default=4, help="photos per cat per batch")
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--name", help="run folder under runs/ (default derived from settings)")


def run_name(stage: str, args: argparse.Namespace, depth: int, suffix: str = "") -> str:
    return (args.name or f"{stage}_{args.backbone}_d{depth}_lr{args.backbone_lr:g}{suffix}"
            + (f"_seed{args.seed}" if args.seed != SEED and not args.name else ""))


def pk_batches(labels: list[int], p: int, k: int, rng: random.Random) -> list[list[int]]:
    """One epoch of P x K batches, about len(labels) / (P*K) of them.

    Each batch has P distinct cats with K photos each (without replacement within
    the batch when the cat has >= K photos).
    """
    by_cat: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_cat.setdefault(label, []).append(index)
    cats = sorted(by_cat)
    batches = []
    for _ in range(len(labels) // (p * k)):
        batch = []
        for cat in rng.sample(cats, p):
            photos = by_cat[cat]
            batch += rng.sample(photos, k) if len(photos) >= k else rng.choices(photos, k=k)
        batches.append(batch)
    return batches


def fit(args: argparse.Namespace, stage: str, rows: list[dict], name: str, depth: int,
        init: Path | None = None, **extra_settings) -> None:
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels_by_cat = label_map(rows)
    labels = [labels_by_cat[row["identity"]] for row in rows]
    images = cached_images(rows, "train")
    augment = train_transform()
    evaluator = Evaluator("pawsadaan", "validation")
    held_out = {r["identity"] for r in evaluator.queries + evaluator.gallery}
    if held_out & set(labels_by_cat):
        raise RuntimeError(f"Training cats overlap validation/decoys: {sorted(held_out & set(labels_by_cat))}")

    model = EmbeddingModel(args.backbone, depth, pretrained=init is None)
    if init is not None:
        bundle = torch.load(init, map_location="cpu", weights_only=False)
        if (bundle["config"]["backbone"], bundle["config"]["trainable_depth"]) != (args.backbone, depth):
            raise RuntimeError(f"--init {init} was trained as {bundle['config']}")
        model.load_state_dict(bundle["state_dict"])
    model.to(device)
    arcface = ArcFace(len(labels_by_cat), scale=args.arcface_scale,
                      margin=args.arcface_margin).to(device)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=backbone_params, lr=args.backbone_lr),
        dict(params=[*model.head.parameters(), *arcface.parameters()], lr=args.head_lr),
    ], weight_decay=args.weight_decay)
    steps_per_epoch = len(rows) // (args.p * args.k)
    total, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup
        else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))))

    settings = dict(vars(args), stage=stage, trainable_depth=depth, name=name,
                    init=str(init) if init else None, train_photos=len(rows),
                    train_cats=len(labels_by_cat), steps_per_epoch=steps_per_epoch,
                    trainable_backbone_params=sum(p.numel() for p in backbone_params),
                    **extra_settings)
    print(json.dumps(settings))
    history, best = [], None
    log = (run_dir / "log.jsonl").open("w", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for batch in pk_batches(labels, args.p, args.k, rng):
            x = torch.stack([augment(images[i]) for i in batch]).to(device)
            y = torch.tensor([labels[i] for i in batch], device=device)
            loss = arcface(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *arcface.parameters()], 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        metrics = evaluator(model, device, ci=False)["all"]
        entry = dict(epoch=epoch, loss=round(float(np.mean(losses)), 4), val_mAP=metrics["mAP"],
                     val_top1=metrics["top1"], val_auc=metrics["verification"]["roc_auc"],
                     val_tpr_at_fpr_1pct=metrics["verification"]["tpr_at_fpr_0.01"],
                     seconds=round(time.time() - started, 1))
        history.append(entry)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        improved = best is None or entry["val_mAP"] > best["val_mAP"]
        if improved:
            best = entry
            save_checkpoint(run_dir / "best.pt", model, epoch=epoch, settings=settings)
        print(f"epoch {epoch:>3}  loss {entry['loss']:.3f}  mAP {entry['val_mAP']:.3f}  "
              f"top1 {entry['val_top1']:.3f}  AUC {entry['val_auc']:.3f}  "
              f"TPR@1% {entry['val_tpr_at_fpr_1pct']:.3f}  {entry['seconds']}s"
              + ("  *" if improved else ""), flush=True)
    log.close()

    best_model = load_checkpoint(run_dir / "best.pt").to(device)
    metrics = evaluator(best_model, device)
    path = write_report(str(run_dir / "best.pt"), best_model, evaluator, metrics, device,
                        training=dict(settings=settings, best_epoch=best["epoch"],
                                      selected_by="PawSaDaan validation mAP (all queries)",
                                      history=history))
    print_table(f"{name} best epoch {best['epoch']} -> {path.name}", metrics)
