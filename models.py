"""Backbones, embedding models, and the ArcFace training head."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# name -> (builder kind, identifier, feature dimension)
BASELINES = {
    "resnet50": ("torchvision", "IMAGENET1K_V2", 2048),
    "dinov2_vits14": ("timm", "vit_small_patch14_dinov2.lvd142m", 384),
    "dinov2_vits14_reg4": ("timm", "vit_small_patch14_reg4_dinov2.lvd142m", 384),
}
EMBEDDING_DIMENSIONS = 512  # contract with pawsadaan-backend


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (module mapping 224px images to pooled features, feature dimension)."""
    kind, identifier, dimension = BASELINES[name]
    if kind == "torchvision":
        from torchvision.models import ResNet50_Weights, resnet50
        # Same weights as the deployed service (pawsadaan-ai-service/ensemble_model.py).
        model = resnet50(weights=ResNet50_Weights[identifier] if pretrained else None)
        model.fc = nn.Identity()
    else:
        import timm
        # num_classes=0 returns the pooled (CLS token) feature; positional
        # embeddings are resized from 518px to the service's 224px input.
        model = timm.create_model(identifier, pretrained=pretrained, num_classes=0, img_size=224)
    return model, dimension


def trainable_backbone_parts(backbone: nn.Module, name: str, depth: int) -> list[nn.Module]:
    """The last `depth` stages that are fine-tuned; everything earlier stays frozen.

    DINOv2: the last `depth` transformer blocks + final norm.
    ResNet50: the last `depth` of layer1..layer4.
    """
    if name == "resnet50":
        return [getattr(backbone, f"layer{i}") for i in range(5 - depth, 5)]
    return [*backbone.blocks[len(backbone.blocks) - depth:], backbone.norm]


class FrozenBaseline(nn.Module):
    """Pretrained backbone with no training: embedding = L2-normalized pooled feature."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.backbone, self.dimension = build_backbone(name)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(images).float(), dim=1)


class EmbeddingModel(nn.Module):
    """Backbone -> linear 512-d projection -> BatchNorm -> L2 normalization."""

    def __init__(self, backbone_name: str, trainable_depth: int, pretrained: bool = True):
        super().__init__()
        self.config = dict(backbone=backbone_name, trainable_depth=trainable_depth,
                           dimensions=EMBEDDING_DIMENSIONS)
        self.backbone, features = build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(nn.Linear(features, EMBEDDING_DIMENSIONS, bias=False),
                                  nn.BatchNorm1d(EMBEDDING_DIMENSIONS))
        self.dimension = EMBEDDING_DIMENSIONS
        self.backbone.requires_grad_(False)
        for part in trainable_backbone_parts(self.backbone, backbone_name, trainable_depth):
            part.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Backbone BatchNorm statistics stay at their pretrained values: batches of
        # 8 cats are too small and too correlated to re-estimate them.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(images)), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin loss (Deng et al., 2019) over the training cats."""

    def __init__(self, classes: int, dimensions: int = EMBEDDING_DIMENSIONS,
                 scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dimensions))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = embeddings @ F.normalize(self.weight, dim=1).T
        target = cosine.gather(1, labels[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(target)
        # Past pi - m, adding the margin would increase cos again; fall back to a
        # linear penalty there (the usual "easy margin off" ArcFace variant).
        margined = torch.where(theta + self.margin < math.pi, torch.cos(theta + self.margin),
                               target - self.margin * math.sin(self.margin))
        logits = cosine.scatter(1, labels[:, None], margined) * self.scale
        return F.cross_entropy(logits, labels)


def save_checkpoint(path: Path, model: EmbeddingModel, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=model.config, state_dict=model.state_dict(), **extra), path)


def load_checkpoint(path: Path) -> EmbeddingModel:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    model = EmbeddingModel(config["backbone"], config["trainable_depth"], pretrained=False)
    model.load_state_dict(bundle["state_dict"])
    model.requires_grad_(False)
    return model.eval()
