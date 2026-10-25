import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import yaml

from network import (
    build_efficientnet_b0,
    load_weights,
    VideoDeepfakeGRU,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class HybridVideoSequenceDataset(Dataset):
    """
    Crash-proof Dataset for training VideoDeepfakeGRU:
    - Loads real video files (.mp4, .avi, etc.) as T-frame sequences.
    - Also supports image folders by synthesizing realistic temporal sequences
      (micro-motion + compression drift for FAKE, smooth consistent motion for REAL),
      ensuring training NEVER fails even if only image data is present initially.
    """
    def __init__(self, root_paths, seq_len=8, is_train=True):
        self.seq_len = seq_len
        self.is_train = is_train
        self.samples = []  # List of (path, label_idx, media_type) where 0=FAKE, 1=REAL

        if isinstance(root_paths, str):
            root_paths = [root_paths]

        class_map = {
            "fake": 0, "fakes": 0, "deepfake": 0, "0": 0,
            "real": 1, "reals": 1, "original": 1, "1": 1,
        }

        for root in root_paths:
            if not os.path.exists(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                folder_name = os.path.basename(dirpath).lower()
                if folder_name not in class_map:
                    continue
                label = class_map[folder_name]
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in VIDEO_EXTS:
                        self.samples.append((fpath, label, "video"))
                    elif ext in IMAGE_EXTS:
                        self.samples.append((fpath, label, "image"))

        self.normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        num_vids = sum(1 for s in self.samples if s[2] == "video")
        num_imgs = sum(1 for s in self.samples if s[2] == "image")
        print(
            f"📂 [{'TRAIN' if is_train else 'VAL'}] Loaded {len(self.samples)} items "
            f"({num_vids} videos, {num_imgs} images -> {self.seq_len}-frame sequences)"
        )

    def __len__(self):
        return len(self.samples)

    def _read_video_sequence(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total > 0:
            indices = np.linspace(0, max(total - 1, 0), self.seq_len, dtype=int)
            target_set = set(indices.tolist())
            idx = 0
            grabbed = {}
            while idx <= max(target_set):
                ret, frame = cap.read()
                if not ret:
                    break
                if idx in target_set and frame is not None:
                    grabbed[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                idx += 1
            for i in indices:
                if i in grabbed:
                    frames.append(grabbed[i])
                elif frames:
                    frames.append(frames[-1].copy())
        cap.release()

        if len(frames) == 0:
            # Fallback blank image if video is unreadable
            frames = [Image.new("RGB", (224, 224), (128, 128, 128))] * self.seq_len

        while len(frames) < self.seq_len:
            frames.append(frames[-1].copy())

        return frames[:self.seq_len]

    def _synthesize_sequence_from_image(self, path, label):
        try:
            base_img = Image.open(path).convert("RGB").resize((240, 240))
        except Exception:
            base_img = Image.new("RGB", (240, 240), (128, 128, 128))

        frames = []
        # Smooth camera pan trajectory
        dx_step = random.uniform(-1.5, 1.5) if self.is_train else 0.5
        dy_step = random.uniform(-1.5, 1.5) if self.is_train else 0.5
        mirror = self.is_train and (random.random() < 0.5)

        for t in range(self.seq_len):
            img = base_img.copy()
            if mirror:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if label == 0 and self.is_train:
                # FAKE (0): simulate temporal deepfake jitter / intermittent blur spike
                jx = int(8 + t * dx_step + random.uniform(-3.0, 3.0))
                jy = int(8 + t * dy_step + random.uniform(-3.0, 3.0))
                if random.random() < 0.35:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
            else:
                # REAL (1): smooth natural temporal continuity
                jx = int(8 + t * dx_step)
                jy = int(8 + t * dy_step)

            jx = max(0, min(16, jx))
            jy = max(0, min(16, jy))
            crop = img.crop((jx, jy, jx + 224, jy + 224))
            frames.append(crop)

        return frames

    def __getitem__(self, idx):
        path, label, media_type = self.samples[idx]
        if media_type == "video":
            pil_frames = self._read_video_sequence(path)
        else:
            pil_frames = self._synthesize_sequence_from_image(path, label)

        tensor_frames = torch.stack([self.normalize(f) for f in pil_frames], dim=0)  # (T, 3, 224, 224)
        return tensor_frames, torch.tensor(label, dtype=torch.long)


def train_gru():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_len = int(cfg.get("video_seq_len", 8))
    batch_size = int(cfg.get("video_batch_size", 8))
    epochs = int(cfg.get("gru_epochs", 12))
    lr = float(cfg.get("lr_head", 3e-4))

    train_paths = cfg.get("video_train_paths", cfg.get("train_paths", ["finetune_data/train"]))
    val_paths = cfg.get("video_val_paths", cfg.get("val_paths", ["finetune_data/val"]))

    train_ds = HybridVideoSequenceDataset(train_paths, seq_len=seq_len, is_train=True)
    val_ds = HybridVideoSequenceDataset(val_paths, seq_len=seq_len, is_train=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset is empty! Check train_paths and val_paths in config.yaml.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load your best 90-95% accurate image model as frozen backbone
    backbone = build_efficientnet_b0(dropout=float(cfg.get("dropout", 0.4)))
    for candidate_weights in ["models/fine_tuned_v1.pt", "models/fine_tuned.pt", cfg.get("base_weights", "models/best_model.pt")]:
        if candidate_weights and os.path.exists(candidate_weights):
            try:
                load_weights(backbone, candidate_weights, map_location="cpu")
                print(f"🌟 Using '{candidate_weights}' as frozen spatial backbone for Bi-GRU!")
                break
            except Exception as e:
                print(f"⚠️ Failed loading '{candidate_weights}': {e}")

    model = VideoDeepfakeGRU(
        backbone=backbone,
        hidden_dim=int(cfg.get("gru_hidden_dim", 256)),
        num_layers=int(cfg.get("gru_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
        freeze_backbone=True,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.05)))

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    save_path = "models/video_gru.pt"

    print(f"\n🚀 Starting Video Bi-GRU + Temporal Attention Training on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()  # Note: Frozen EfficientNet stays in .eval() automatically!
        train_loss, train_correct, train_total = 0.0, 0, 0

        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss / max(1, train_total):.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best Video Bi-GRU model -> '{save_path}' (Val Acc: {best_val_acc * 100:.2f}%)")

    print(f"\n🎉 Training Complete! Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"✅ Trained Video Bi-GRU saved at: '{save_path}'")


if __name__ == "__main__":
    train_gru()