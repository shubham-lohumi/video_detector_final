import math
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPVisionModel

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TemporalAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled, weights.squeeze(-1)


class CLIPTemporalHead(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        d_model=512,
        nhead=8,
        num_layers=4,
        ff_dim=1024,
        dropout=0.30,
        num_classes=2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        in_dim = feature_dim * 3

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = TemporalAttentionPool(d_model)
        self.out_norm = nn.LayerNorm(d_model * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, frame_features):
        norm_f = F.normalize(frame_features, p=2, dim=-1)

        delta1 = torch.zeros_like(norm_f)
        if norm_f.size(1) > 1:
            delta1[:, 1:, :] = norm_f[:, 1:, :] - norm_f[:, :-1, :]

        delta2 = torch.zeros_like(norm_f)
        if norm_f.size(1) > 2:
            delta2[:, 2:, :] = delta1[:, 2:, :] - delta1[:, 1:-1, :]

        combined = torch.cat([frame_features, delta1 * 10.0, delta2 * 10.0], dim=-1)
        x = self.input_proj(combined)
        x = self.pos_enc(x)
        x = self.transformer(x)

        attn_ctx, attn_weights = self.attn_pool(x)
        mean_ctx = torch.mean(x, dim=1)
        video_repr = self.out_norm(torch.cat([attn_ctx, mean_ctx], dim=-1))
        logits = self.classifier(video_repr)
        return logits, attn_weights


class CLIPEnvDetector:
    """
    Global Environment Video Specialist using frozen CLIP ViT-B/16 (768-dim) +
    trained 4-Layer Temporal Transformer with calibrated temperature and FPR threshold.
    """

    def __init__(self, checkpoint_path="models/clip_temporal_env.pt", device=None):
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.is_ready = False
        self.temperature = 1.0
        self.threshold = 0.50
        self.seq_len = 16

        self.clip_tf = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

        if not os.path.exists(checkpoint_path):
            print(
                f"⚠️ '{checkpoint_path}' not found. Environment CLIP stream disabled."
            )
            self.vision = None
            self.head = None
            return

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        clip_name = ckpt.get("clip_model_name", "openai/clip-vit-base-patch16")
        arch_cfg = ckpt.get("arch_config", {})

        self.vision = CLIPVisionModel.from_pretrained(clip_name).to(self.device)
        self.vision.eval()

        self.head = CLIPTemporalHead(**arch_cfg).to(self.device)
        self.head.load_state_dict(ckpt["state_dict"], strict=True)
        self.head.eval()

        self.temperature = float(ckpt.get("temperature", 1.0))
        self.threshold = float(ckpt.get("optimal_threshold", 0.50))
        self.seq_len = int(ckpt.get("seq_len", 16))
        self.is_ready = True
        print(
            f"✅ Loaded CLIP-Temporal Environment Specialist: '{checkpoint_path}' "
            f"(T={self.temperature:.2f}, Thresh={self.threshold:.3f})"
        )

    def predict_frames(self, pil_frames):
        """
        Runs aspect-preserving CLIP + Temporal Transformer on full uncropped frames.
        Returns dict with calibrated fake probability, attention weights, and scene flicker.
        """
        if not self.is_ready or len(pil_frames) == 0:
            return None

        batch = torch.stack([self.clip_tf(f) for f in pil_frames], dim=0).to(
            self.device
        )  # (T, 3, 224, 224)

        with torch.no_grad():
            feats = self.vision(pixel_values=batch).pooler_output  # (T, 768)
            seq_feats = feats.unsqueeze(0)  # (1, T, 768)

            # Cosine scene drift across consecutive full frames
            norm_f = F.normalize(feats, p=2, dim=-1)
            if norm_f.size(0) >= 2:
                scene_drift = float(
                    torch.mean(1.0 - torch.sum(norm_f[1:] * norm_f[:-1], dim=-1)).item()
                )
            else:
                scene_drift = 0.0

            logits, attn_weights = self.head(seq_feats)
            probs = torch.softmax(logits / self.temperature, dim=1)
            env_fake_prob = float(probs[0, 0].item())

        return {
            "env_fake_prob": env_fake_prob,
            "env_threshold": self.threshold,
            "env_attn_weights": [
                round(float(w), 4) for w in attn_weights[0].cpu().tolist()
            ],
            "scene_drift": round(min(max(scene_drift * 3.0, 0.0), 1.0), 4),
        }
