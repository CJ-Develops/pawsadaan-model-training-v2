# PawSaDaan Model Training v2

Retraining workspace for the PawSaDaan cat re-identification embedding model,
built on **PawSaDaan AI Training Collated v2**. Succeeds `../pawsadaan-model-training`
(ResNet50, v1 data), which stays untouched until this model replaces the one
deployed in `../pawsadaan-ai-service`.

## Layout

```
config.py            paths, seed, preprocessing contract (matches the AI service)
data.py              manifest row selection, image loading, transforms, Dataset
freeze_manifests.py  copies + locks the v2 split manifests, verifies every image
models.py            backbones, 512-d embedding model, ArcFace loss, checkpoints
evaluate.py          retrieval + verification metrics on validation (test needs --allow-test)
training.py          shared ArcFace training loop (P x K batches, per-epoch validation)
train_stage2.py      Hello Street Cat pre-training (tested; not used, see below)
train_stage3.py      PawSaDaan fine-tuning (best epoch by validation mAP)
compare.py           paired cat-bootstrap comparison of two models
calibrate_and_export.py  threshold on validation, test once, export .pt + .json
exports/             exported service bundles (.pt gitignored, .json committed)
manifests/           frozen manifest copy + FROZEN.json (hashes and counts)
runs/                checkpoints and logs (gitignored)
results/             metric reports (committed)
```

Photos are read in place from `../PawSaDaan Source Folder`; they are never copied here.

## Setup

```powershell
py -3.13 -m venv venv
venv\Scripts\activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Data splits

Splits are identity-disjoint: no cat appears in more than one split.

| Set | Images | Cats | Use |
|---|---|---|---|
| Hello Street Cat train | 1885 | 38 | Stage 2 pre-training |
| Hello Street Cat val / test | 248 / 618 | 8 / 8 | Stage 2 sanity checks |
| PawSaDaan train | 932 (+105 supplemental) | 53 | Stage 3 fine-tuning |
| PawSaDaan validation | 196 | 12 | checkpoint selection, threshold calibration |
| PawSaDaan test | 223 | 12 | final report, once |
| PawSaDaan limited identities | 323 | 44 | gallery decoys only, never trained on |

`python freeze_manifests.py --check` verifies the frozen manifests and every
referenced image. `data.py` refuses to load if the frozen manifests were edited.
Re-freezing (`--force`) changes the splits under every existing result; do it
only when new cats are added upstream, and re-run all baselines afterwards.

## Evaluate

```powershell
python evaluate.py --model dinov2_vits14                  # PawSaDaan validation + 44 decoy cats
python evaluate.py --model resnet50 --dataset hsc         # Hello Street Cat validation
```

`--model` also takes a trained checkpoint (`runs/<name>/best.pt`). Writes `results/<dataset>_<split>_<model>.json` (metrics with cat-level bootstrap
95% CIs, protocol notes, manifest hashes) and prints a summary table.

## Train and compare

```powershell
python train_stage3.py --backbone dinov2_vits14 --backbone-lr 3e-5   # ~8 min on a GTX 1660 SUPER
python compare.py runs/<A>/best.pt runs/<B>/best.pt
```

Backbone choice (2026-09-25): DINOv2 ViT-S/14 beat the ResNet50 control after identical
training — validation mAP 0.808 ± 0.011 vs 0.694 ± 0.014 over 3 seeds. Hello Street Cat
pre-training (Stage 2) lowered validation mAP by ~0.02 on both seeds tested, so it is not
used (details in CLAUDE.md).

## Plan

1. ~~Workspace, environment, frozen manifests~~
2. Photograph the 24 validation/test cats on a new day (improves cross-session evaluation)
3. ~~`evaluate.py`: frozen ImageNet ResNet50 vs. frozen DINOv2 ViT-S/14 baselines~~
4. ~~Stage 3 backbone comparison: DINOv2 vs. ResNet50 control~~ (DINOv2)
5. ~~Stage 2: Hello Street Cat pre-training~~ (tested; did not help, dropped)
6. ~~Calibrate threshold on validation, report test once~~ (threshold 0.5165; test recall 0.655 at 0.74% FPR)
7. ~~Export `.pt` + `.json`~~ (`exports/dinov2-vits14-arcface-pawsadaan-v1`)
8. Load the bundle in `pawsadaan-ai-service`; re-embed all stored cats
