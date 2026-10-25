import io
import os
import random
from collections import Counter

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from network import IMAGENET_MEAN, IMAGENET_STD


class RandomJPEGCompression:
    def __init__(self, quality_min=60, quality_max=95, p=0.35):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Locked mapping: 0 = FAKE, 1 = REAL
        class_mapping = {"fake": 0, "real": 1}

        for class_name, label in class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(is_train=True):
    if is_train:
        # Keep geometric transforms mild so frequency artifacts survive.
        return transforms.Compose([
            transforms.Resize((232, 232)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
            RandomJPEGCompression(quality_min=60, quality_max=95, p=0.35),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(paths, is_train=True):
    transform = build_transforms(is_train=is_train)
    datasets = [CustomImageDataset(p, transform=transform) for p in paths if os.path.exists(p)]
    if not datasets:
        raise FileNotFoundError(f"No dataset folders found in: {paths}")

    combined = ConcatDataset(datasets)
    counts = Counter()
    for ds in datasets:
        counts.update(label for _, label in ds.samples)

    split = "train" if is_train else "val"
    print(
        f"📊 {split}: {len(combined)} images | "
        f"FAKE={counts.get(0, 0)} | REAL={counts.get(1, 0)}"
    )
    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        print("⚠️ One class is empty. Check fake/ and real/ folder names (lowercase).")
    return combined
