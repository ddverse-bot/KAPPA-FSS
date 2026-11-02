import os
import random
import numpy as np
import torch

from .losses import kappa_loss
from .metrics import evaluate_episode_metrics
from .data import MEAN, STD

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_episode(model, episode, shots, device):
    support = episode["support"][:shots]
    support_images = [sample[0].to(device) for sample in support]
    support_masks = [sample[1].to(device) for sample in support]
    query_image, query_mask = episode["query"]
    output = model(support_images, support_masks, query_image.to(device))
    target = query_mask.unsqueeze(0).to(device)
    return output, target

def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "assouad": 0.0}
    for index, episode in enumerate(loader):
        shots = random.choice(config.shots)
        output, target = run_episode(model, episode, shots, device)
        loss, components = kappa_loss(output["logits"], target, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        totals["total"] += loss.item()
        for key, value in components.items():
            totals[key] += value
    count = max(len(loader), 1)
    return {key: value / count for key, value in totals.items()}

@torch.no_grad()
def evaluate(model, loader, shots, config, device):
    model.eval()
    values = {"mIoU": [], "Dice": [], "BoundaryIoU": [], "HD95": []}
    for episode in loader:
        output, target = run_episode(model, episode, shots, device)
        metrics = evaluate_episode_metrics(output["logits"], target, config.threshold)
        for key, value in metrics.items():
            values[key].append(value)
    return {
        key: float(np.nanmean(value)) if len(value) and not np.all(np.isnan(value)) else float("nan")
        for key, value in values.items()
    } | {"N": len(loader)}

def save_checkpoint(path, model, optimizer, config, epoch, dataset):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.__dict__,
        "epoch": epoch,
        "dataset": dataset,
    }, path)
