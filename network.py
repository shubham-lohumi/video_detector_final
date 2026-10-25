import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_FEATURE_BLOCKS = 9  # EfficientNet-B0 features indices 0..8
CLASS_NAMES = {0: "FAKE", 1: "REAL"}  # locked mapping used everywhere
EFFICIENTNET_B0_FEAT_DIM = 1280


def build_efficientnet_b0(dropout=0.4, num_classes=2):
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def _strip_prefix(key):
    for prefix in ("model.model.", "model.", "module.", "net."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state dict or checkpoint dict, got {type(payload)}")

    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    elif "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]

    cleaned = OrderedDict()
    for key, value in payload.items():
        if any(key.startswith(skip) for skip in ("loss_fn.", "criterion.")):
            continue
        cleaned[_strip_prefix(key)] = value
    return cleaned


def load_weights(model, path, map_location="cpu"):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    state_dict = extract_state_dict(payload)
    result = model.load_state_dict(state_dict, strict=False)

    missing = [k for k in result.missing_keys if "num_batches_tracked" not in k]
    unexpected = result.unexpected_keys
    loaded = len(state_dict) - len(unexpected)

    print(f"✅ Loaded weights from '{path}' ({loaded} tensors)")
    if missing:
        print(f"   Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"   Unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    classifier_missing = any(k.startswith("classifier.") for k in missing)
    return not classifier_missing


def freeze_feature_blocks(model, freeze_until_block):
    """Freeze features[0 .. freeze_until_block-1]. Classifier always stays trainable."""
    freeze_until_block = max(0, min(int(freeze_until_block), NUM_FEATURE_BLOCKS))
    for name, param in model.features.named_parameters():
        block_num = int(name.split(".", 1)[0])
        param.requires_grad = block_num >= freeze_until_block
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"🧊 Freeze blocks 0..{freeze_until_block - 1} | "
        f"trainable params: {trainable:,} | frozen: {frozen:,}"
    )


def param_groups(model, lr_backbone, lr_head, weight_decay):
    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []

    for name, param in model.named_parameters():
        target_decay = decay_head if name.startswith("classifier") else decay_backbone
        target_no_decay = no_decay_head if name.startswith("classifier") else no_decay_backbone
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            target_no_decay.append(param)
        else:
            target_decay.append(param)

    return [
        {"params": decay_backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": decay_head, "lr": lr_head, "weight_decay": weight_decay},
        {"params": no_decay_head, "lr": lr_head, "weight_decay": 0.0},
    ]


# =====================================================================
# VIDEO DEEPFAKE DETECTION: Bi-GRU + Temporal Attention Architecture
# =====================================================================

def extract_backbone_features(backbone, x):
    """
    Extracts 1280-dim pooled features and 2-class logits from EfficientNet-B0.
    Input x: (N, 3, 224, 224)
    Returns: features (N, 1280), logits (N, 2)
    """
    feat_maps = backbone.features(x)
    pooled = backbone.avgpool(feat_maps)
    features = torch.flatten(pooled, 1)
    logits = backbone.classifier(features)
    return features, logits


class TemporalAttention(nn.Module):
    """
    Attention mechanism over video frames so suspicious/manipulated frames
    receive higher weight than normal frames.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, gru_out):
        # gru_out: (B, T, hidden_dim)
        scores = self.attn(gru_out)              # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # (B, T, 1)
        context = torch.sum(weights * gru_out, dim=1)  # (B, hidden_dim)
        return context, weights.squeeze(-1)


class VideoDeepfakeGRU(nn.Module):
    """
    Spatial-Temporal Deepfake Detector using Frozen EfficientNet-B0 + Bi-GRU + Attention.
    Captures both per-frame artifacts (f_t) and inter-frame flicker/motion anomalies (f_t - f_{t-1}).
    """
    def __init__(
        self,
        backbone=None,
        feature_dim=EFFICIENTNET_B0_FEAT_DIM,
        proj_dim=512,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        num_classes=2,
        freeze_backbone=True,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_efficientnet_b0(dropout=0.4, num_classes=num_classes)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Input to projection is [f_t, delta_f_t] -> feature_dim * 2 (2560)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional GRU for temporal sequence modeling
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2  # 512 for bidirectional
        self.attention = TemporalAttention(gru_out_dim)
        self.temporal_norm = nn.LayerNorm(gru_out_dim)

        self.temporal_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # Learnable gate balancing spatial backbone logits & temporal Bi-GRU logits
        self.temporal_gate = nn.Parameter(torch.tensor(0.0))

    def train(self, mode=True):
        """
        CRITICAL: Keep frozen EfficientNet BatchNorm layers in .eval() mode
        during GRU training so running stats never corrupt the 90-95% image accuracy!
        """
        super().train(mode)
        if self.freeze_backbone and self.backbone is not None:
            self.backbone.eval()
        return self

    def forward_features(self, frame_features, frame_logits=None):
        """
        Runs Bi-GRU + Temporal Attention on pre-extracted frame features.
        frame_features: (B, T, 1280)
        frame_logits: Optional (B, T, 2)
        Returns: final_logits (B, 2), attn_weights (B, T)
        """
        # Normalize features along channel dim for stable temporal delta calculation
        norm_feats = F.normalize(frame_features, p=2, dim=-1)

        # Temporal difference (Delta features: captures frame-to-frame face jitter/flicker)
        delta_feats = torch.zeros_like(norm_feats)
        if norm_feats.size(1) > 1:
            delta_feats[:, 1:, :] = norm_feats[:, 1:, :] - norm_feats[:, :-1, :]

        # Combine raw features + scaled temporal delta
        combined = torch.cat([frame_features, delta_feats * 10.0], dim=-1)  # (B, T, 2560)
        proj = self.input_proj(combined)                                   # (B, T, 512)

        gru_out, _ = self.gru(proj)                                        # (B, T, 512)
        attn_context, attn_weights = self.attention(gru_out)               # (B, 512), (B, T)
        mean_context = torch.mean(gru_out, dim=1)                          # (B, 512)

        video_repr = self.temporal_norm(attn_context + mean_context)       # (B, 512)
        temporal_logits = self.temporal_classifier(video_repr)             # (B, 2)

        if frame_logits is not None:
            # Attention-weighted spatial logits + mean spatial logits
            spatial_attn_logits = torch.sum(frame_logits * attn_weights.unsqueeze(-1), dim=1)
            spatial_mean_logits = torch.mean(frame_logits, dim=1)
            spatial_logits = 0.5 * (spatial_attn_logits + spatial_mean_logits)

            alpha = torch.sigmoid(self.temporal_gate)
            final_logits = (1.0 - alpha) * spatial_logits + alpha * temporal_logits
        else:
            final_logits = temporal_logits

        return final_logits, attn_weights

    def forward(self, x):
        """
        Full forward pass from video frames tensor.
        x: (B, T, C, H, W)
        Returns: final_logits (B, 2)
        """
        b, t, c, h, w = x.shape
        flat_x = x.contiguous().view(b * t, c, h, w)

        if self.freeze_backbone:
            with torch.no_grad():
                feats, logits = extract_backbone_features(self.backbone, flat_x)
        else:
            feats, logits = extract_backbone_features(self.backbone, flat_x)

        feats = feats.view(b, t, -1)
        logits = logits.view(b, t, -1)

        final_logits, _ = self.forward_features(feats, logits)
        return final_logits