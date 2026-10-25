import os

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from datasets.hybrid_loader import load_datasets
from lightning_modules.detector import DeepfakeDetector
from network import build_efficientnet_b0, freeze_feature_blocks, load_weights


def make_loader(dataset, batch_size, shuffle, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )


def export_best_checkpoint(best_ckpt, cfg):
    if not best_ckpt or not os.path.exists(best_ckpt):
        print("⚠️ No best checkpoint found; skipping .pt export.")
        return

    os.makedirs("models", exist_ok=True)
    pt_output = "models/fine_tuned.pt"
    print(f"\n🔄 Exporting '{best_ckpt}' -> '{pt_output}'")

    backbone = build_efficientnet_b0(dropout=cfg.get("dropout", 0.4))
    best_model = DeepfakeDetector.load_from_checkpoint(
        best_ckpt,
        model=backbone,
        lr_head=cfg.get("lr_head", 3e-4),
        lr_backbone=cfg.get("lr_backbone", 3e-5),
        weight_decay=cfg.get("weight_decay", 1e-4),
        label_smoothing=cfg.get("label_smoothing", 0.05),
        unfreeze_epoch=cfg.get("phase1_epochs", 4),
        freeze_until_block=cfg.get("freeze_until_block", 4),
        max_epochs=cfg.get("num_epochs", 20),
    )
    torch.save(best_model.model.state_dict(), pt_output)
    print(f"🎉 Saved inference weights: {pt_output}")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    train_dataset = load_datasets(cfg["train_paths"], is_train=True)
    val_dataset = load_datasets(cfg["val_paths"], is_train=False)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Train or val set is empty. Run prepare_dataset.py first.")

    pin_memory = torch.cuda.is_available()
    batch_size = int(cfg.get("batch_size", 32))
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, pin_memory=pin_memory)

    backbone = build_efficientnet_b0(dropout=cfg.get("dropout", 0.4))
    checkpoint_path = cfg.get("base_weights", "models/best_model.pt")
    if os.path.exists(checkpoint_path):
        try:
            load_weights(backbone, checkpoint_path, map_location="cpu")
        except Exception as exc:
            print(f"⚠️ Could not load '{checkpoint_path}': {exc}")
            print("   Continuing from ImageNet backbone + random classifier.")
    else:
        print(f"⚠️ '{checkpoint_path}' not found. Starting from ImageNet weights.")

    freeze_feature_blocks(backbone, freeze_until_block=9)

    model = DeepfakeDetector(
        backbone,
        lr_head=float(cfg.get("lr_head", 3e-4)),
        lr_backbone=float(cfg.get("lr_backbone", 3e-5)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        label_smoothing=float(cfg.get("label_smoothing", 0.05)),
        unfreeze_epoch=int(cfg.get("phase1_epochs", 4)),
        freeze_until_block=int(cfg.get("freeze_until_block", 4)),
        max_epochs=int(cfg.get("num_epochs", 20)),
    )

    os.makedirs("models", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="best_model_finetuned",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=int(cfg.get("early_stopping_patience", 6)),
        mode="max",
        min_delta=0.001,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = pl.Trainer(
        max_epochs=int(cfg.get("num_epochs", 20)),
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=int(cfg.get("log_every_n_steps", 10)),
        gradient_clip_val=1.0,
        precision=precision,
        deterministic=False,
    )

    print("🚀 Starting two-phase fine-tuning (head first, then later blocks)...")
    trainer.fit(model, train_loader, val_loader)
    export_best_checkpoint(checkpoint_callback.best_model_path, cfg)


if __name__ == "__main__":
    main()
