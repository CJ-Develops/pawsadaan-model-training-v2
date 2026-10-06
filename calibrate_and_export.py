"""Calibrate the match threshold on validation, report test once, export the bundle.

1. Threshold (validation only): one reference photo per gallery cat, like the
   app's one photo per CatProfile; the score with the highest TPR at pairwise
   FPR <= --target-fpr (the same rule as the deployed refinement_v2 bundle).
2. Test (only with --allow-test, and only once: refuses if a final test report
   exists): full evaluate.py metrics on the 12 test cats + 44 decoys, and the
   calibrated threshold applied unchanged.
3. Export (only with --allow-test): exports/<version>.pt (weights_only-loadable:
   state_dict + metadata) and exports/<version>.json (same metadata, no weights),
   in the format of pawsadaan-ai-service/weights/.

Without --allow-test it is a dry run: validation calibration + preprocessing
parity check only; nothing is written.

    python calibrate_and_export.py runs/stage3_dinov2_vits14_d4_lr3e-05/best.pt
    python calibrate_and_export.py runs/stage3_dinov2_vits14_d4_lr3e-05/best.pt --allow-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import roc_curve
from torchvision.models import ResNet50_Weights

from config import FROZEN_RECORD, PREPROCESS, PROJECT_ROOT, RESULTS_DIR
from data import eval_transform, load_image
from evaluate import Evaluator, evaluate, load_model, print_table, single_reference_columns
from models import BASELINES

EXPORT_DIR = PROJECT_ROOT / "exports"
MODEL_VERSION_BASE = "dinov2-vits14-arcface-pawsadaan-v1"
ARCHITECTURE = "dinov2-vits14-linear-bn-l2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_reference(evaluator: Evaluator, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(similarities to one reference photo per gallery cat, same-cat mask)."""
    q_ids = np.array([r["identity"] for r in evaluator.queries])
    g_ids = np.array([r["identity"] for r in evaluator.gallery])
    columns = single_reference_columns(g_ids)
    return sim[:, columns], q_ids[:, None] == g_ids[columns][None, :]


def at_threshold(sim: np.ndarray, same: np.ndarray, threshold: float) -> dict:
    """Pairwise confusion counts plus per-report rates, for score >= threshold."""
    predicted = sim >= threshold
    tp, fp = int((predicted & same).sum()), int((predicted & ~same).sum())
    fn, tn = int((~predicted & same).sum()), int((~predicted & ~same).sum())
    wrong_best = np.where(same, -np.inf, sim).max(axis=1)
    return dict(
        threshold=threshold, true_positive=tp, false_positive=fp, true_negative=tn,
        false_negative=fn, recall=round(tp / (tp + fn), 4), false_positive_rate=round(fp / (fp + tn), 5),
        precision=round(tp / max(1, tp + fp), 4),
        # Per query photo: is its own cat's reference above the threshold, and does
        # any other cat's reference also pass (a "possible match" false alarm)?
        report_catch_rate=round(float((sim[same] >= threshold).mean()), 4),
        report_false_alarm_rate=round(float((wrong_best >= threshold).mean()), 4),
        other_cats_per_report=int((~same).sum(axis=1)[0]),
    )


def calibrate(sim: np.ndarray, same: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, thresholds = roc_curve(same.ravel(), sim.ravel())
    ok = np.flatnonzero(fpr <= target_fpr)
    return float(thresholds[ok[np.argmax(tpr[ok])]])


@torch.no_grad()
def service_parity(model: torch.nn.Module, rows: list[dict]) -> float:
    """Lowest cosine between embeddings from our eval_transform and the AI service's
    preprocessing (ResNet50_Weights.IMAGENET1K_V2.transforms()) on the same photos."""
    ours, service = eval_transform(), ResNet50_Weights.IMAGENET1K_V2.transforms()
    model = model.cpu().eval()
    images = [load_image(row) for row in rows]
    a = model(torch.stack([ours(image) for image in images]))
    b = model(torch.stack([service(image) for image in images]))
    return float((a * b).sum(dim=1).min())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--allow-test", action="store_true",
                        help="evaluate the test split (once) and write the export bundle")
    args = parser.parse_args()
    if args.checkpoint.name in BASELINES or not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    run = args.checkpoint.resolve().parent.name
    test_report = RESULTS_DIR / f"final_test_{run}.json"
    if args.allow_test and test_report.exists():
        parser.error(f"{test_report.name} exists: the test split was already reported for this model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(args.checkpoint)).to(device)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    validation = Evaluator("pawsadaan", "validation")
    val_sim, val_same = single_reference(validation, validation.similarities(model, device))
    threshold = calibrate(val_sim, val_same, args.target_fpr)
    val_at = at_threshold(val_sim, val_same, threshold)
    print(f"threshold {threshold:.4f} (validation, one reference per cat, pairwise FPR <= "
          f"{args.target_fpr:g})\nvalidation at threshold: {json.dumps(val_at)}")
    parity = service_parity(model, validation.queries[:16])
    print(f"AI-service preprocessing parity (min cosine, 16 photos): {parity:.5f}")
    if parity < 0.999:
        raise RuntimeError("Service preprocessing differs from training preprocessing")
    if not args.allow_test:
        print("dry run: pass --allow-test to evaluate the test split once and export")
        return

    model.to(device)
    test = Evaluator("pawsadaan", "test")
    test_full_sim = test.similarities(model, device)
    test_metrics = evaluate(test_full_sim, test.queries, test.gallery)
    test_sim, test_same = single_reference(test, test_full_sim)
    test_at = at_threshold(test_sim, test_same, threshold)
    print_table(f"{run} on pawsadaan TEST (reported once)", test_metrics)
    print(f"test at validation threshold: {json.dumps(test_at)}")

    frozen = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))
    config = bundle["config"]
    settings = bundle["settings"]
    metadata = dict(
        format_version=1, architecture=ARCHITECTURE, dimensions=config["dimensions"],
        preprocess=PREPROCESS, model_version_base=MODEL_VERSION_BASE,
        backbone=dict(timm_model="vit_small_patch14_dinov2.lvd142m", img_size=224,
                      num_classes=0, pooling="cls_token", feature_dimensions=384,
                      timm_version=str(timm.__version__), torch_version=str(torch.__version__)),
        head="Linear(384, 512, bias=False) -> BatchNorm1d(512) -> L2 normalize",
        source_checkpoint=f"runs/{run}/best.pt",
        source_checkpoint_sha256=sha256_file(args.checkpoint),
        manifest_sha256=frozen["manifest_sha256"],
        match_threshold=threshold,
        calibration=dict(split="val", gallery="one reference per cat",
                         rule=f"Maximum TPR at pairwise FPR <= {args.target_fpr:g} on validation",
                         verification=val_at, service_preprocess_parity_min_cosine=parity),
        test_metrics=dict(split="test", note="reported once; nothing was tuned on it",
                          at_threshold=test_at, all=test_metrics["all"],
                          cross_session=test_metrics.get("cross_session")),
        training=dict(stage="stage3 only (no Hello Street Cat pre-training)",
                      best_epoch=bundle["epoch"], loss="ArcFace s=30 m=0.3",
                      backbone_lr=settings["backbone_lr"], trainable_depth=config["trainable_depth"],
                      train_photos=settings["train_photos"], train_cats=settings["train_cats"],
                      seed=settings["seed"]),
        experimental=True,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    EXPORT_DIR.mkdir(exist_ok=True)
    weights_path = EXPORT_DIR / f"{MODEL_VERSION_BASE}.pt"
    state_dict = {key: value.cpu() for key, value in bundle["state_dict"].items()}
    torch.save(dict(metadata, state_dict=state_dict), weights_path)
    torch.load(weights_path, map_location="cpu", weights_only=True)  # service loads it this way
    metadata["bundle_sha256"] = sha256_file(weights_path)
    (EXPORT_DIR / f"{MODEL_VERSION_BASE}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    test_report.write_text(json.dumps(dict(model=run, threshold=threshold, validation_at_threshold=val_at,
                                           test_at_threshold=test_at, test_metrics=test_metrics),
                                      indent=2) + "\n", encoding="utf-8")
    print(f"-> {weights_path.relative_to(PROJECT_ROOT)} (+ .json), {test_report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
