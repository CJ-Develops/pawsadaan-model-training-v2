"""Retrieval + verification evaluation of an embedding model on one split.

Protocol (PawSaDaan):
  * Query/gallery come from the evaluated split's primary photos.
    - Cats whose rows carry eval_role (known capture sessions) use it: gallery =
      earlier session, query = later session. Reported as "cross_session".
    - Other cats: a deterministic per-cat split of their photos (hash-ordered,
      first half gallery, rest query). Reported as "session_unknown"; these
      photos may come from the same day, so these numbers are optimistic.
  * The 44 limited-identity cats are added to the gallery as decoys.
  * Retrieval: top-1/top-5 rank cats by their best-matching gallery photo (like
    the app's top-5 list); mAP is over individual gallery photos.
  * Verification: every query photo vs. every gallery photo is one pair (the app
    compares one photo with one cat photo); TPR at FPR <= 1% and 0.1%, ROC AUC.
  * "single_reference": the same, but each gallery cat keeps only its first
    gallery photo, matching the app's one-photo-per-CatProfile setup.

Hello Street Cat (--dataset hsc) uses its manifest eval_role (front = gallery,
top = query) and has no decoys.

--model is a frozen baseline name or a trained checkpoint (runs/<name>/best.pt).
Thresholds printed here are informational; calibrate_and_export.py sets the
deployed one on validation. The test split needs --allow-test (report it once).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from config import FROZEN_RECORD, RESULTS_DIR, SEED
from data import cached_images, distractor_rows, normalize_uint8, stage2_rows, stage3_rows
from models import BASELINES, FrozenBaseline, load_checkpoint

FPR_TARGETS = (0.01, 0.001)
BOOTSTRAP_ROUNDS = 1000


def _order_key(row: dict) -> str:
    return hashlib.sha256(f"{SEED}:{row['image_path']}".encode()).hexdigest()


def build_protocol(dataset: str, split: str) -> tuple[list[dict], list[dict], dict]:
    """Return (query rows, gallery rows, notes). Each row gains a 'group' key."""
    rows = stage2_rows(split) if dataset == "hsc" else stage3_rows(split)
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["identity"]].append(row)

    queries, gallery, dropped = [], [], 0
    for identity in sorted(by_cat):
        photos = sorted(by_cat[identity], key=_order_key)
        roled = [row for row in photos if row["eval_role"] in ("gallery", "query")]
        if roled:
            group = "cross_session" if dataset == "pawsadaan" else "front_to_top"
            dropped += len(photos) - len(roled)
            for row in roled:
                (gallery if row["eval_role"] == "gallery" else queries).append({**row, "group": group})
        else:
            if dataset == "hsc":
                raise RuntimeError(f"HSC {split} cat {identity} has no eval_role rows")
            half = max(1, len(photos) // 2)
            gallery += [{**row, "group": "session_unknown"} for row in photos[:half]]
            queries += [{**row, "group": "session_unknown"} for row in photos[half:]]

    decoys = []
    if dataset == "pawsadaan":
        decoys = [{**row, "group": "decoy"} for row in distractor_rows()]
        overlap = {r["identity"] for r in rows} & {r["identity"] for r in decoys}
        if overlap:
            raise RuntimeError(f"Decoy cats overlap the evaluated split: {sorted(overlap)}")
    notes = dict(split_cats=len(by_cat), decoy_cats=len({r["identity"] for r in decoys}),
                 roled_cats_rows_without_role_ignored=dropped)
    return queries, gallery + decoys, notes


def load_model(spec: str) -> torch.nn.Module:
    """A frozen baseline name (see models.BASELINES) or a trained checkpoint path."""
    return FrozenBaseline(spec) if spec in BASELINES else load_checkpoint(Path(spec))


def model_label(spec: str) -> str:
    """Report name: the baseline name, or the run folder of a checkpoint."""
    return spec if spec in BASELINES else Path(spec).resolve().parent.name


@torch.no_grad()
def embed(model: torch.nn.Module, images: torch.Tensor, device: str,
          batch_size: int = 64) -> np.ndarray:
    """Embed cached uint8 eval images (see data.cached_images) in eval mode.

    fp32 on purpose: fp16 autocast makes ResNet50 output NaN at batch >= 32 on
    the GTX 1660 SUPER (cuDNN), and inference here is only ~800 photos.
    """
    was_training = model.training
    model.eval()
    out = [model(normalize_uint8(images[start:start + batch_size].to(device))).float().cpu()
           for start in range(0, len(images), batch_size)]
    model.train(was_training)
    return torch.cat(out).numpy()


class Evaluator:
    """Holds one split's query/gallery photos in memory so models can be re-scored cheaply."""

    def __init__(self, dataset: str = "pawsadaan", split: str = "validation"):
        self.dataset, self.split = dataset, split
        self.queries, self.gallery, self.notes = build_protocol(dataset, split)
        self.query_images = cached_images(self.queries, "eval")
        self.gallery_images = cached_images(self.gallery, "eval")

    def similarities(self, model: torch.nn.Module, device: str) -> np.ndarray:
        query_emb = embed(model, self.query_images, device)
        gallery_emb = embed(model, self.gallery_images, device)
        if not (np.isfinite(query_emb).all() and np.isfinite(gallery_emb).all()):
            raise RuntimeError("Embeddings contain NaN/inf")
        return query_emb @ gallery_emb.T

    def __call__(self, model: torch.nn.Module, device: str, ci: bool = True) -> dict:
        return evaluate(self.similarities(model, device), self.queries, self.gallery, ci)


def _bootstrap_ci(values: np.ndarray, cats: np.ndarray) -> list[float]:
    """95% CI of the mean, resampling cats (not photos) with replacement."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(cats)
    per_cat = [values[cats == cat] for cat in unique]
    means = [np.concatenate([per_cat[i] for i in rng.integers(0, len(unique), len(unique))]).mean()
             for _ in range(BOOTSTRAP_ROUNDS)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def verification(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Pair-level ROC AUC and TPR at fixed FPR targets."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    result = dict(roc_auc=round(float(roc_auc_score(labels, scores)), 4),
                  positive_pairs=int(labels.sum()), negative_pairs=int((~labels).sum()))
    for target in FPR_TARGETS:
        ok = np.flatnonzero(fpr <= target)
        best = ok[np.argmax(tpr[ok])]
        result[f"tpr_at_fpr_{target:g}"] = round(float(tpr[best]), 4)
        result[f"threshold_at_fpr_{target:g}"] = round(float(thresholds[best]), 4)
    return result


def per_query(sim: np.ndarray, q_ids: np.ndarray, g_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query top-1 / top-5 hits (cats ranked by their best photo) and average precision."""
    matches = q_ids[:, None] == g_ids[None, :]
    if not matches.any(axis=1).all():
        raise RuntimeError("A query cat has no gallery photo")
    cats = np.unique(g_ids)
    cat_sim = np.stack([sim[:, g_ids == cat].max(axis=1) for cat in cats], axis=1)
    true_score = cat_sim[np.arange(len(q_ids)), np.searchsorted(cats, q_ids)]
    rank = (cat_sim > true_score[:, None]).sum(axis=1)  # 0 = correct cat ranked first
    return dict(top1=(rank < 1).astype(float), top5=(rank < 5).astype(float),
                mAP=np.array([average_precision_score(m, s) for m, s in zip(matches, sim)]))


def score(sim: np.ndarray, q_ids: np.ndarray, g_ids: np.ndarray, ci: bool = True) -> dict:
    """Metrics for query rows vs. gallery columns of a cosine-similarity matrix."""
    result = dict(queries=len(q_ids), query_cats=len(np.unique(q_ids)),
                  gallery_photos=len(g_ids), gallery_cats=len(np.unique(g_ids)))
    for name, values in per_query(sim, q_ids, g_ids).items():
        result[name] = round(float(values.mean()), 4)
        if ci:
            result[f"{name}_ci95"] = _bootstrap_ci(values, q_ids)
    result["verification"] = verification((q_ids[:, None] == g_ids[None, :]).ravel(), sim.ravel())
    return result


def single_reference_columns(g_ids: np.ndarray) -> np.ndarray:
    """One reference photo per gallery cat: the first in the (hash-stable) gallery order."""
    first = {}
    for index, identity in enumerate(g_ids):
        first.setdefault(identity, index)
    return np.array(sorted(first.values()))


def evaluate(sim: np.ndarray, queries: list[dict], gallery: list[dict], ci: bool = True) -> dict:
    q_ids = np.array([r["identity"] for r in queries])
    g_ids = np.array([r["identity"] for r in gallery])
    q_groups = np.array([r["group"] for r in queries])
    single = single_reference_columns(g_ids)

    report = {}
    for group in ["all", *sorted(set(q_groups))]:
        mask = np.ones(len(q_ids), bool) if group == "all" else q_groups == group
        report[group] = score(sim[mask], q_ids[mask], g_ids, ci)
        report[group]["single_reference"] = score(sim[mask][:, single], q_ids[mask],
                                                  g_ids[single], ci)
    return report


def write_report(spec: str, model: torch.nn.Module, evaluator: Evaluator, metrics: dict,
                 device: str, **extra) -> Path:
    frozen = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))
    name = model_label(spec)
    report = dict(
        model=name, model_kind="frozen_pretrained" if spec in BASELINES else "trained",
        model_config=getattr(model, "config", None), embedding_dimensions=model.dimension,
        dataset=evaluator.dataset, split=evaluator.split, protocol=evaluator.notes,
        manifest_sha256=frozen["manifest_sha256"], device=device,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra, metrics=metrics,
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{evaluator.dataset}_{evaluator.split}_{name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def print_table(title: str, metrics: dict) -> None:
    print(title)
    print(f"{'group':<18}{'queries':>8}{'top1':>8}{'top5':>8}{'mAP':>8}{'TPR@1%':>8}{'AUC':>8}"
          f"{'1ref top1':>11}{'1ref TPR@1%':>13}")
    for group, m in metrics.items():
        v, s = m["verification"], m["single_reference"]
        print(f"{group:<18}{m['queries']:>8}{m['top1']:>8.3f}{m['top5']:>8.3f}{m['mAP']:>8.3f}"
              f"{v['tpr_at_fpr_0.01']:>8.3f}{v['roc_auc']:>8.3f}"
              f"{s['top1']:>11.3f}{s['verification']['tpr_at_fpr_0.01']:>13.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True,
                        help=f"baseline ({', '.join(sorted(BASELINES))}) or checkpoint path")
    parser.add_argument("--dataset", default="pawsadaan", choices=("pawsadaan", "hsc"))
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--allow-test", action="store_true",
                        help="required for --split test; report the test split once, at the end")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("the test split is for the final report only; pass --allow-test")
    if args.model not in BASELINES and not Path(args.model).is_file():
        parser.error(f"not a baseline name or checkpoint file: {args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args.dataset, args.split)
    model = load_model(args.model).to(device)
    metrics = evaluator(model, device)
    path = write_report(args.model, model, evaluator, metrics, device)
    print_table(f"{model_label(args.model)} on {args.dataset} {args.split}  ->  "
                f"{path.relative_to(RESULTS_DIR.parent)}", metrics)


if __name__ == "__main__":
    main()
