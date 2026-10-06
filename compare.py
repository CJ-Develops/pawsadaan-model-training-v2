"""Paired comparison of two embedding models on the same validation queries.

Both models embed the same query/gallery photos (evaluate.Evaluator). The
difference B - A of each metric is bootstrapped by resampling query cats with
replacement (the same resample for both models), giving a 95% CI of the gap and
how often B comes out ahead. A gap whose CI excludes 0 is treated as real.

    python compare.py runs/<A>/best.pt runs/<B>/best.pt
    python compare.py resnet50 dinov2_vits14

Writes results/compare_<dataset>_<split>_<A>_vs_<B>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import RESULTS_DIR, SEED
from evaluate import (Evaluator, load_model, model_label, per_query, single_reference_columns,
                      verification)
from models import BASELINES

BOOTSTRAP_ROUNDS = 1000


def paired_metrics(sim: np.ndarray, q_ids: np.ndarray, g_ids: np.ndarray,
                   rows: np.ndarray) -> dict[str, float]:
    """Metrics over the (possibly repeated) query rows `rows`."""
    sim, q = sim[rows], q_ids[rows]
    single = single_reference_columns(g_ids)
    hits = per_query(sim, q, g_ids)
    pairs = verification((q[:, None] == g_ids[None, :]).ravel(), sim.ravel())
    single_pairs = verification((q[:, None] == g_ids[single][None, :]).ravel(),
                                sim[:, single].ravel())
    return {"mAP": hits["mAP"].mean(), "top1": hits["top1"].mean(),
            "roc_auc": pairs["roc_auc"], "tpr_at_fpr_1pct": pairs["tpr_at_fpr_0.01"],
            "single_reference_tpr_at_fpr_1pct": single_pairs["tpr_at_fpr_0.01"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", help="baseline name or checkpoint path (reference)")
    parser.add_argument("b", help="baseline name or checkpoint path (challenger)")
    parser.add_argument("--dataset", default="pawsadaan", choices=("pawsadaan", "hsc"))
    args = parser.parse_args()
    for spec in (args.a, args.b):
        if spec not in BASELINES and not Path(spec).is_file():
            parser.error(f"not a baseline name or checkpoint file: {spec}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(args.dataset, "validation")
    sims = {}
    for spec in (args.a, args.b):
        model = load_model(spec).to(device)
        sims[spec] = evaluator.similarities(model, device)
        del model
        torch.cuda.empty_cache()

    q_ids = np.array([r["identity"] for r in evaluator.queries])
    g_ids = np.array([r["identity"] for r in evaluator.gallery])
    cats = np.unique(q_ids)
    rows_of = {cat: np.flatnonzero(q_ids == cat) for cat in cats}
    everything = np.arange(len(q_ids))
    a_full = paired_metrics(sims[args.a], q_ids, g_ids, everything)
    b_full = paired_metrics(sims[args.b], q_ids, g_ids, everything)

    rng = np.random.default_rng(SEED)
    gaps = {name: [] for name in a_full}
    for _ in range(BOOTSTRAP_ROUNDS):
        rows = np.concatenate([rows_of[cat] for cat in rng.choice(cats, len(cats))])
        a = paired_metrics(sims[args.a], q_ids, g_ids, rows)
        b = paired_metrics(sims[args.b], q_ids, g_ids, rows)
        for name in gaps:
            gaps[name].append(b[name] - a[name])

    a_name, b_name = model_label(args.a), model_label(args.b)
    result = {}
    print(f"B - A on {args.dataset} validation, {len(cats)} query cats, "
          f"{BOOTSTRAP_ROUNDS} paired cat resamples\n  A = {a_name}\n  B = {b_name}")
    print(f"{'metric':<34}{'A':>8}{'B':>8}{'B-A':>8}{'95% CI':>20}{'P(B>A)':>8}")
    for name, values in gaps.items():
        low, high = np.percentile(values, [2.5, 97.5])
        result[name] = dict(a=round(float(a_full[name]), 4), b=round(float(b_full[name]), 4),
                            gap=round(float(b_full[name] - a_full[name]), 4),
                            gap_ci95=[round(float(low), 4), round(float(high), 4)],
                            b_ahead_fraction=round(float(np.mean(np.array(values) > 0)), 3))
        r = result[name]
        print(f"{name:<34}{r['a']:>8.3f}{r['b']:>8.3f}{r['gap']:>+8.3f}"
              f"{f'[{low:+.3f}, {high:+.3f}]':>20}{r['b_ahead_fraction']:>8.2f}")

    path = RESULTS_DIR / f"compare_{args.dataset}_validation_{a_name}_vs_{b_name}.json"
    path.write_text(json.dumps(dict(a=a_name, b=b_name, dataset=args.dataset, split="validation",
                                    query_cats=len(cats), bootstrap_rounds=BOOTSTRAP_ROUNDS,
                                    metrics=result), indent=2) + "\n", encoding="utf-8")
    print(f"-> {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
