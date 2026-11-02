import matplotlib.pyplot as plt
import torch

from .data import MEAN, STD
from .metrics import binarize
from .engine import run_episode

@torch.no_grad()
def show_episode(model, episode, shots, device, threshold=0.5):
    model.eval()
    output, target = run_episode(model, episode, shots, device)
    prediction = binarize(output["logits"], threshold)[0, 0].cpu().numpy()
    target_array = target[0, 0].cpu().numpy()
    image = episode["query"][0]
    image = (image * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(image)
    axes[0].set_title("Query")
    axes[1].imshow(target_array, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title(f"{shots}-shot prediction")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure
