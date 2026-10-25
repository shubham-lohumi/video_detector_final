import os
import random
import shutil

import torch
from PIL import Image, ImageEnhance
from facenet_pytorch import MTCNN
from tqdm import tqdm

BASE_RAW_DIR = r"C:\Users\DELL\Desktop\downloads"
OUTPUT_DIR = "finetune_data"

FAKE_CONFIG = [
    ("fake_kudh_ki", None, 3),
    ("AI-Generated Images", None, 1),
    ("training_fake", None, 1),
    ("ai_hai", 4000, 1),
]

REAL_CONFIG = [
    ("real_kudh_ki", None, 3),
    ("Real Images", None, 1),
    ("training_real", None, 1),
    ("real", 4000, 1),
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mtcnn = MTCNN(
    image_size=224,
    margin=24,
    keep_all=False,
    post_process=False,
    select_largest=True,
    thresholds=[0.5, 0.6, 0.6],
    device=device,
)


def get_all_images(folder_path):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    img_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(exts):
                img_list.append(os.path.join(root, filename))
    return img_list


def augment_face(face_img, multiplier):
    variants = [face_img]
    if multiplier > 1:
        variants.append(face_img.transpose(Image.FLIP_LEFT_RIGHT))
    if multiplier > 2:
        variants.append(ImageEnhance.Brightness(face_img).enhance(1.1))
    return variants


def extract_faces_from_sources(config_list):
    faces = []

    for folder_name, max_limit, multiplier in config_list:
        folder_path = os.path.join(BASE_RAW_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: folder not found -> {folder_path}")
            continue

        all_imgs = get_all_images(folder_path)
        random.shuffle(all_imgs)
        if max_limit is not None and len(all_imgs) > max_limit:
            all_imgs = all_imgs[:max_limit]

        print(f"\n📂 Scanning {len(all_imgs)} images from '{folder_name}'...")
        valid_from_folder = 0
        for img_path in tqdm(all_imgs, desc=f"Cropping {folder_name}"):
            try:
                raw_img = Image.open(img_path).convert("RGB")
                face = mtcnn(raw_img)
                if face is None:
                    continue
                face_array = face.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
                face_img = Image.fromarray(face_array)
                faces.append((face_img, multiplier))
                valid_from_folder += 1
            except Exception:
                continue

        print(f"✅ Extracted {valid_from_folder} unique faces from '{folder_name}'")

    return faces


def save_balanced_splits(fake_faces, real_faces):
    min_len = min(len(fake_faces), len(real_faces))
    random.shuffle(fake_faces)
    random.shuffle(real_faces)
    fake_faces = fake_faces[:min_len]
    real_faces = real_faces[:min_len]

    print(f"\n⚖️ Balancing unique faces: {min_len} FAKE and {min_len} REAL")

    for label_name, face_list in [("fake", fake_faces), ("real", real_faces)]:
        train_dir = os.path.join(OUTPUT_DIR, "train", label_name)
        val_dir = os.path.join(OUTPUT_DIR, "val", label_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        split_idx = int(len(face_list) * 0.85)
        train_items = face_list[:split_idx]
        val_items = face_list[split_idx:]

        train_count = 0
        for face_img, multiplier in train_items:
            for variant in augment_face(face_img, multiplier):
                variant.save(os.path.join(train_dir, f"{label_name}_{train_count:05d}.jpg"), quality=95)
                train_count += 1

        for idx, (face_img, _) in enumerate(val_items):
            face_img.save(os.path.join(val_dir, f"{label_name}_{idx:05d}.jpg"), quality=95)

        print(f"📁 {label_name.upper()} -> Train: {train_count} | Val: {len(val_items)}")


if __name__ == "__main__":
    random.seed(42)

    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Cleaning old '{OUTPUT_DIR}' folder...")
        shutil.rmtree(OUTPUT_DIR)

    print("🚀 Starting dataset preparation...")
    fake_list = extract_faces_from_sources(FAKE_CONFIG)
    real_list = extract_faces_from_sources(REAL_CONFIG)
    save_balanced_splits(fake_list, real_list)
    print("\n🎉 Dataset ready. Train-only copies were augmented; val stays clean.")
