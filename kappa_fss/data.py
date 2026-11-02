import os
import re
import glob
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_SUFFIX = re.compile(r"(_segmentation|_seg|_mask|_gt|_groundtruth)$", re.IGNORECASE)

def _find_named_dir(root, name):
    for current, dirs, _ in os.walk(root):
        if os.path.basename(current).lower() == name.lower():
            return current
    return None

def _all_images(root):
    return sorted(
        os.path.join(current, name)
        for current, _, names in os.walk(root)
        for name in names
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    )

def _base_id(path):
    return MASK_SUFFIX.sub("", Path(path).stem)

def discover_kvasir_pairs(root):
    image_dir = _find_named_dir(root, "images")
    mask_dir = _find_named_dir(root, "masks")
    if not image_dir or not mask_dir:
        raise FileNotFoundError("Kvasir-SEG requires directories named images and masks.")
    images = _all_images(image_dir)
    masks = {_base_id(path): path for path in _all_images(mask_dir)}
    pairs = [(path, masks[_base_id(path)]) for path in images if _base_id(path) in masks]
    if not pairs:
        raise RuntimeError("No matching Kvasir-SEG image and mask pairs were found.")
    return pairs

def discover_isic_pairs(root):
    all_files = _all_images(root)
    mask_files = []
    image_files = []
    for path in all_files:
        name = Path(path).name.lower()
        parent = Path(path).parent.name.lower()
        is_mask = (
            any(token in name for token in ("segmentation", "_mask", "_gt", "groundtruth"))
            or parent in ("gt", "masks", "mask", "groundtruth")
        )
        (mask_files if is_mask else image_files).append(path)
    masks = {}
    for path in mask_files:
        masks.setdefault(_base_id(path), path)
    pairs = [(path, masks[_base_id(path)]) for path in image_files if _base_id(path) in masks]
    unique = {}
    for image_path, mask_path in pairs:
        unique[os.path.realpath(image_path)] = (image_path, mask_path)
    pairs = sorted(unique.values())
    if not pairs:
        raise RuntimeError("No matching ISIC 2016 image and segmentation-mask pairs were found.")
    return pairs

def discover_pairs(dataset, root):
    dataset = dataset.lower()
    if dataset == "kvasir":
        return discover_kvasir_pairs(root)
    if dataset == "isic2016":
        return discover_isic_pairs(root)
    raise ValueError(f"Unsupported dataset: {dataset}")

def load_pair(image_path, mask_path, image_size=400):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    image = torch.from_numpy(np.asarray(image).copy()).float().permute(2, 0, 1) / 255.0
    image = (image - MEAN) / STD
    mask = torch.from_numpy((np.asarray(mask).copy() > 127).astype(np.float32)).unsqueeze(0)
    return image, mask

class EpisodicSegmentationDataset(Dataset):
    def __init__(self, pairs, n_episodes, max_shot=5, image_size=400, seed=42):
        if len(pairs) < max_shot + 1:
            raise ValueError(f"At least {max_shot + 1} paired samples are required.")
        self.pairs = list(pairs)
        self.n_episodes = n_episodes
        self.max_shot = max_shot
        self.image_size = image_size
        rng = random.Random(seed)
        self.episodes = [
            (chosen[:max_shot], chosen[max_shot])
            for chosen in (rng.sample(range(len(self.pairs)), max_shot + 1) for _ in range(n_episodes))
        ]

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        support_indices, query_index = self.episodes[index]
        support = [load_pair(*self.pairs[i], self.image_size) for i in support_indices]
        query = load_pair(*self.pairs[query_index], self.image_size)
        return {"support": support, "query": query}

def episode_collate(batch):
    return batch[0]

def make_loaders(pairs, config, episode_counts=None):
    import random
    rng = random.Random(config.seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_train = int(len(indices) * config.train_fraction)
    n_val = int(len(indices) * config.val_fraction)
    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs = [pairs[i] for i in indices[n_train:n_train + n_val]]
    test_pairs = [pairs[i] for i in indices[n_train + n_val:]]
    counts = episode_counts or {"train": config.train_episodes, "val": config.val_episodes, "test": config.test_episodes}
    datasets = {
        "train": EpisodicSegmentationDataset(train_pairs, counts["train"], config.max_shot, config.image_size, config.seed),
        "val": EpisodicSegmentationDataset(val_pairs, counts["val"], config.max_shot, config.image_size, config.seed + 1),
        "test": EpisodicSegmentationDataset(test_pairs, counts["test"], config.max_shot, config.image_size, config.seed + 2),
    }
    loaders = {
        key: DataLoader(value, batch_size=1, shuffle=(key == "train"), collate_fn=episode_collate)
        for key, value in datasets.items()
    }
    return datasets, loaders
