# PawSaDaan Model Training v2 — working context

Retraining the cat re-identification embedding model for PawSaDaan, a TNVR
cat-monitoring system (Capstone, Group 10, BSIT 3-2, PUP CICS; Django backend +
FastAPI embedding service). The model maps a cat photo to an L2-normalized
embedding; two photos match when cosine similarity ≥ a calibrated threshold.
README.md has setup commands and the split table; this file records decisions,
state, and what to do next.

## Sibling folders (all under `C:\Users\John Arcilla\Documents`)

| Folder | Role |
|---|---|
| `pawsadaan-model-training-v2/` | **this repo** — new training + evaluation |
| `pawsadaan-model-training/` | OLD workspace (ResNet50, v1 data, not in git). Archive only: do not modify or delete until the new model is deployed — the service's current weights came from its `checkpoints/`. |
| `pawsadaan-ai-service/` | FastAPI service that loads the deployed model (`ensemble_model.py`, `model_loader.py`, `weights/*.pt` + `*.json`) |
| `pawsadaan-backend/` | Django app; stores embeddings. Embedding dimension is a contract with it (see `PawSaDaan_Model_Training_Setup.md` in the Source Folder). |
| `PawSaDaan Source Folder/` | Raw + collated photos. Dataset in use: `PawSaDaan AI Training Collated v2`. Also holds `hellostreetcat/`. Older `PawSaDaan AI Training Collated` (v1) is superseded. |

## Why retrain (decided with the user, 2026-09-23 to 09-25)

- Deployed models are ImageNet ResNet50-based: `pawsadaan_refinement_v2` (512-d) and an
  exploratory 50/50 ensemble (2560-d, "HelloStreetCat 50/50; administrator review required").
- refinement_v2's own metadata: recall 0.65 at ~1% FPR, measured on the same validation
  split used to pick its threshold → optimistic; ~35% of true matches missed.
- Old checkpoints were trained on a different manifest than v2, so they cannot be scored
  fairly on v2's validation/test cats (possible leakage). Honest numbers require retraining.

## Decisions

- **Backbone: DINOv2 ViT-S/14 — confirmed 2026-09-25** (`timm`
  `vit_small_patch14_dinov2.lvd142m`, `img_size=224`; 384-d CLS → 512-d projection). Won the
  frozen baseline and, decisively, the trained ResNet50 control (see "Backbone comparison").
  Train the last 4 blocks + norm + head; backbone LR 3e-5 (1e-5–1e-4 all fine), head LR 1e-3.
  ResNet50 control code stays (`--backbone resnet50`) for the write-up; no need to re-run it.
- MegaDescriptor was considered and set aside (likely CC BY-NC license, wildlife-focused).
- **Pipeline: DINOv2 init → Stage 3 PawSaDaan fine-tune → threshold on validation → report
  test once. Stage 2 (Hello Street Cat pre-training) was tested and DROPPED on 2026-09-25:** it
  made Stage 3 worse on both seeds (see "Stage 2 ablation"). `train_stage2.py` and
  `train_stage3.py --init` stay for the write-up; don't re-run unless HSC data changes.
- **Loss/batching:** metric learning (ArcFace preferred), P×K batches (e.g. 8 cats × 4 photos),
  batch ~32 (6 GB VRAM). 512-d output.
- **Mixed precision is suspect on this GPU:** fp16 autocast makes ResNet50 output NaN at batch
  ≥ 32 on the GTX 1660 SUPER (cuDNN 9.7; fine with cuDNN disabled; DINOv2 fp16 was fine), and a
  quick timing showed no fp16 speedup. evaluate.py runs fp32. Re-test (with warm-up) before
  using AMP in training; always check losses/embeddings for NaN.
- **Photos are never copied** into this repo; they are read from the Source Folder via the
  manifest `image_path` (relative to `SOURCE_ROOT` in `config.py`).
- **Commits:** the user commits; don't commit unless asked.

## Done

1. Workspace + git repo created (initial files staged, **not committed** — user reviews/commits).
2. `venv/` (Python 3.13): torch 2.7.1+cu128, torchvision 0.22.1, timm 1.0.30, pandas,
   scikit-learn, tqdm. CUDA verified on GTX 1660 SUPER (6 GB).
3. v2 manifests frozen into `manifests/` with hashes in `FROZEN.json`; all referenced images
   exist and match their SHA-256. `.gitattributes` marks `manifests/*` as `-text` so Git does
   not alter line endings (would break the hash check).
4. `data.py` smoke test: all 4,592 images decode to 3×224×224, 0 failures; train /
   validation / test / decoy identities are disjoint.
5. `models.py` (frozen baselines only: `resnet50` = torchvision IMAGENET1K_V2, same as the
   service; `dinov2_vits14`, `dinov2_vits14_reg4` via timm, CLS token) and `evaluate.py`.
   Protocol: roled cats (Dolly, Snow / Kring, Tyler) use eval_role → group `cross_session`;
   others hash-split per cat (`sha256(SEED:image_path)`, first half gallery) → group
   `session_unknown` (likely same-day, optimistic). Gallery + 44 decoy cats (323 photos).
   Top-k ranks cats by best gallery photo; mAP over photos; verification = all query×gallery
   photo pairs. `single_reference` = one gallery photo per cat (the app compares against one
   CatProfile photo). Bootstrap CIs resample cats. `--split test` requires `--allow-test`.
6. `data.cached_images` (decode once → uint8 tensors in `runs/cache/`; eval cache reproduces
   eval_transform exactly), `models.EmbeddingModel` (backbone → Linear 512 → BN → L2; backbone
   BN frozen) + `ArcFace` (s=30, m=0.3), `train_stage3.py` (P×K 8×4, AdamW, 1-epoch warm-up +
   cosine, 30 epochs, fp32, best epoch by validation mAP → `runs/<name>/best.pt` + report),
   `compare.py` (paired cat-bootstrap of B − A). Grid: `runs/grid_stage3.sh`, `runs/seeds_stage3.sh`.

## Frozen baselines (PawSaDaan validation, 2026-09-25; 96 queries, 12 cats, 423 gallery photos / 56 cats)

| model | top-1 [95% CI] | top-5 | mAP | TPR@FPR1% | AUC | cross-session top-1 (15 q, 2 cats) | 1-ref top-1 |
|---|---|---|---|---|---|---|---|
| resnet50 | 0.906 [0.84, 0.96] | 0.979 | 0.549 | 0.393 | 0.898 | 0.800 | 0.500 |
| dinov2_vits14 | 0.906 [0.75, 0.99] | 0.990 | 0.599 | 0.440 | 0.941 | 0.533 | 0.552 |
| dinov2_vits14_reg4 | 0.885 [0.71, 0.99] | 1.000 | 0.579 | 0.403 | 0.935 | 0.400 | 0.583 |

HSC validation (8 cats, front→top): resnet50 top-1 0.517 / AUC 0.733; dinov2_vits14 0.652 / 0.851.
Reading: DINOv2 is better on the ranking-free metrics (mAP, AUC, TPR) and on HSC; top-1 is a
tie within wide CIs. Cross-session (2 cats) is too small to decide anything. Frozen verification
is weak for both (TPR@1% ≈ 0.4) — expected before metric-learning training. Decision: proceed
with `dinov2_vits14` (plain; reg4 showed no gain).

## Backbone comparison — Stage 3 only, no Stage 2 (PawSaDaan validation, 2026-09-25)

Same grid for both: backbone LR {1e-5, 3e-5, 1e-4, 3e-4}, seed 42. Best validation mAP per LR:
DINOv2 0.814 / **0.821** / 0.793 / 0.748; ResNet50 (layer4) 0.655 / 0.660 / 0.691 / **0.709**.
Every DINOv2 LR beat every ResNet50 LR. Best LR re-run with seeds 1, 2 (mean ± SD of 3 seeds):

| backbone (LR) | mAP | top-1 | TPR@FPR1% | AUC | 1-ref top-1 | 1-ref TPR@1% | cross-session mAP |
|---|---|---|---|---|---|---|---|
| DINOv2 (3e-5) | **0.808 ± 0.011** | 0.955 ± 0.051 | **0.744 ± 0.023** | 0.958 ± 0.025 | 0.764 | **0.753** | 0.443 |
| ResNet50 (3e-4) | 0.694 ± 0.014 | 0.938 ± 0.028 | 0.623 ± 0.020 | 0.940 ± 0.023 | 0.625 | 0.580 | 0.313 |

Paired test, seed-42 runs (`results/compare_*`): DINOv2 − ResNet50 mAP +0.112 [+0.074, +0.159],
TPR@1% +0.146 [+0.057, +0.188], 1-ref TPR@1% +0.229 [+0.011, +0.330]; top-1 tie. Caveats: LR and
epoch picked on the same validation cats (both models equally optimistic); cross-session = 2 cats.
Both overfit after ~15 epochs (train loss → ~0.001, val mAP drifts down) — consider 20 epochs or
stronger augmentation. Seed-to-seed spread (~0.01–0.02 mAP) is the noise floor for later ablations:
compare runs over ≥ 2 seeds, not single runs.

## Stage 2 ablation — DINOv2, PawSaDaan validation, 2026-09-25 (`runs/stage2_ablation.sh`)

Stage 2 = 20 epochs ArcFace on HSC train (38 cats, 1,885 photos), epoch picked on PawSaDaan
validation mAP; then Stage 3 from it (`--init`). Best-epoch validation mAP:

| seed | Stage 3 only (reference) | Stage 2 → Stage 3, LR 3e-5 (matched) | Stage 2 → Stage 3, LR 1e-5 | Stage 2 alone |
|---|---|---|---|---|
| 42 | **0.821** (TPR@1% 0.766) | 0.799 (0.731) | 0.783 (0.685) | 0.686 (epoch 7/20) |
| 1 | **0.802** (0.745) | 0.778 (0.711) | 0.778 (0.712) | 0.659 (epoch 2/20) |

Stage 2 lowered mAP by ~0.02 on both seeds (≈ 2× the seed spread), also TPR@1% and 1-ref top-1.
Stage 2 alone peaks within a few epochs, then HSC-specific learning hurts PawSaDaan transfer:
HSC photos are fixed-camera, often top-down; PawSaDaan are phone photos. Decision: drop it.

## Final model — exported 2026-09-25 (TEST SPLIT HAS BEEN USED; do not re-evaluate to tune)

`exports/dinov2-vits14-arcface-pawsadaan-v1.pt` (+ `.json`; .pt gitignored) from
`runs/stage3_dinov2_vits14_d4_lr3e-05/best.pt` (seed 42, epoch 13) via `calibrate_and_export.py`.
Architecture `dinov2-vits14-linear-bn-l2`, 512-d, service preprocessing (parity cosine 1.0000).
Bundle loads with `weights_only=True`; rebuilt-from-bundle embeddings match the checkpoint.
**match_threshold = 0.5165** (user chose ≈0.52): validation, one reference per cat, max TPR at
pairwise FPR ≤ 1% (same rule as refinement_v2).

| split (one reference per cat, threshold 0.5165) | recall | pairwise FPR | precision | reports falsely flagged vs 55 other cats |
|---|---|---|---|---|
| validation (calibration) | 0.740 | 0.97% | 0.58 | 47.9% |
| **test (once)** | **0.655** | **0.74%** | 0.62 | 27.4% |

Test, full protocol (113 queries, 12 cats + 44 decoys): top-1 0.956, top-5 1.000, mAP 0.791,
TPR@FPR1% 0.630, AUC 0.975; cross-session (Kring, Tyler; 20 queries) top-1 1.000, mAP 0.854.
Validation → test drop (recall 0.74 → 0.65) is the expected selection optimism. Not directly
comparable to refinement_v2's 0.648 recall (its own, optimistic validation split, other data).
Full report: `results/final_test_stage3_dinov2_vits14_d4_lr3e-05.json`.

**vs. the deployed models on the same v2 test cats** (2026-09-25; checked: no v2 validation/test
cat was in the old models' training set; 9/12 v2 test cats were old test cats; each model at its
own threshold, one reference per cat):

| model | top-1 | mAP | TPR@FPR1% | recall @ own thr | pairwise FPR @ own thr | reports falsely flagged |
|---|---|---|---|---|---|---|
| refinement_v2 (deployed, thr 0.5449) | 0.796 | 0.600 | 0.434 | 0.620 | 2.19% | 54.0% |
| ensemble 50/50 (thr 0.8128) | 0.770 | 0.564 | 0.382 | 0.177 | 0.05% | 2.7% |
| **new DINOv2 v1 (thr 0.5165)** | **0.956** | **0.791** | **0.630** | **0.655** | **0.74%** | **27.4%** |

Paired cat-bootstrap on test, new − refinement_v2: mAP [+0.10, +0.31], top-1 [+0.03, +0.33],
TPR@1% [+0.08, +0.36] (all P(new>old) = 1.00). refinement_v2's threshold no longer holds its
promised 1% FPR on new cats (2.2%). This supersedes the "cannot be scored fairly" note above.

**Backbone control on test** (scored after DINOv2 was locked and exported; nothing tuned;
`results/final_test_backbone_control_resnet50_vs_dinov2.json`): identically trained ResNet50
(`runs/stage3_resnet50_d1_lr0.0003`, own validation threshold 0.6089) vs DINOv2 v1 — top-1
0.805 vs 0.956, mAP 0.539 vs 0.792, TPR@FPR1% 0.409 vs 0.630, AUC 0.904 vs 0.975, recall at own
threshold 0.389 vs 0.655 (pairwise FPR 0.69% vs 0.74%), cross-session top-1 0.45 vs 1.00.
Paired CIs (DINOv2 − ResNet50): mAP [+0.13, +0.38], top-1 [+0.05, +0.29], TPR@1% [+0.12, +0.35].

## Data facts (from the frozen manifests)

- `stage2_rows(split)`: HSC rows with `include=1`. Train 1885 imgs/38 cats, val 248/8, test 618/8.
  HSC val/test rows carry `eval_role` gallery (front view) / query (top view).
- `stage3_rows(split, include_supplemental=False)`: PawSaDaan `primary` rows (supplemental
  adds `supplemental_for_primary`, 105 train photos). Train 932/53, val 196/12, test 223/12.
- `distractor_rows()`: 323 imgs / 44 limited-identity cats (`split=auxiliary_holdout`) —
  gallery decoys only, never train on them.
- Cross-session evaluation is thin: only Dolly, Snow (validation) and Kring, Tyler (test) have
  two known capture sessions; their rows carry `eval_role` gallery/query. Everything else has
  `capture_group=unknown`. 73 cats are listed in `pawsadaan_session_collection_needed.csv`.
- Manifest paths are relative to `SOURCE_ROOT`. Preprocessing (`config.PREPROCESS`) matches the
  AI service: resize 232 → center crop 224, ImageNet mean/std, EXIF transpose.

## Next steps

1. **(User, optional but recommended)** photograph the 24 validation/test cats on a new day,
   re-run upstream `create_split_manifests.py`, then `python freeze_manifests.py --force`.
   Do this before baselines so first numbers use the better test set.
2. ~~`evaluate.py` + frozen baselines~~, ~~backbone comparison~~ (done; see tables). If the
   manifests are re-frozen, re-run the baselines and the DINOv2 Stage 3 reference runs.
3. ~~Stage 2 ablation~~ (done: dropped; see table). Shared loop now lives in `training.py`.
4. ~~`--supplemental` test~~ (done, 2026-09-25, `runs/supplemental.sh`): +105 photos (1,037 vs
   932) gave validation mAP 0.800 vs 0.821 (seed 42) and 0.800 vs 0.802 (seed 1); TPR@1% 0.706 /
   0.736 vs 0.766 / 0.745. No gain → train on primary photos only. Final Stage 3 recipe:
   DINOv2 ViT-S/14, last 4 blocks, backbone LR 3e-5, 30 epochs (best epoch), primary photos,
   no Stage 2. Candidate for export: `runs/stage3_dinov2_vits14_d4_lr3e-05/best.pt` (seed 42).
   Untried: fewer epochs / stronger augmentation (overfits after ~15 epochs).
5. ~~`calibrate_and_export.py`~~ (done; see "Final model"). Refuses a second test run for the
   same checkpoint (`results/final_test_<run>.json` exists).
6. Service integration (next): in `pawsadaan-ai-service`, add `timm` to requirements and a
   loader branch for architecture `dinov2-vits14-linear-bn-l2` (timm model from bundle metadata,
   `pretrained=False`, Linear 384→512 no bias → BatchNorm1d → L2; `strict=True`), copy the bundle
   into `weights/`, set `MODEL_WEIGHTS_PATH`. Dimension stays 512 (Django unchanged), but
   `model_version` changes, so **re-embed every stored cat photo** (backend refuses mixed versions).

## Rules that protect result honesty

- Never tune anything (threshold, epochs, hyperparameters) on the test split.
- Never train on validation, test, or decoy cats.
- Never edit files in `manifests/` by hand; re-freeze with `--force` and re-run baselines.
- Split by cat (already enforced) — never by individual photo.
