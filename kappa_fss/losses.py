import torch
import torch.nn.functional as F

from .modules import assouad_map

def dice_loss(logits, target, epsilon=1e-6):
    probabilities = torch.sigmoid(logits)
    numerator = 2 * (probabilities * target).sum() + epsilon
    denominator = probabilities.sum() + target.sum() + epsilon
    return 1 - numerator / denominator

def assouad_weighted_loss(logits, target, beta=1.0, small_scales=(1, 2, 4), large_scales=(16, 32, 64), temperature=0.1):
    complexity = assouad_map(target, small_scales, large_scales, temperature)
    band = (complexity > 0).float()
    if band.sum() < 1:
        return torch.zeros((), device=logits.device)
    minimum = complexity.min()
    maximum = complexity.max()
    normalized = (complexity - minimum) / (maximum - minimum + 1e-6)
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = (1 + beta * normalized) * band * pixel_loss
    return weighted.sum() / band.sum().clamp(min=1)

def kappa_loss(logits, target, config):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    assouad = assouad_weighted_loss(
        logits,
        target,
        beta=config.beta_assouad,
        small_scales=config.assouad_small_scales,
        large_scales=config.assouad_large_scales,
        temperature=config.assouad_tau,
    )
    total = bce + dice + config.lambda_assouad * assouad
    return total, {"bce": bce.detach().item(), "dice": dice.detach().item(), "assouad": assouad.detach().item()}
