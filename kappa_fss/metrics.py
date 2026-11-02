import math
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

def binarize(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).float()

def overlap_metrics(prediction, target, epsilon=1e-6):
    intersection = float((prediction * target).sum())
    union = float(((prediction + target) > 0).sum())
    dice = (2 * intersection + epsilon) / (float(prediction.sum() + target.sum()) + epsilon)
    iou = (intersection + epsilon) / (union + epsilon)
    return iou, dice

def boundary_band(mask, width):
    outside = distance_transform_edt(1 - mask)
    inside = distance_transform_edt(mask)
    return ((outside <= width) | (inside <= width)).astype(np.float32)

def boundary_iou(prediction, target, diagonal_fraction=0.02):
    diagonal = math.hypot(*prediction.shape)
    width = max(1, int(round(diagonal_fraction * diagonal)))
    pred_band = boundary_band(prediction, width)
    target_band = boundary_band(target, width)
    intersection = np.logical_and(pred_band, target_band).sum()
    union = np.logical_or(pred_band, target_band).sum()
    return float(intersection / union) if union > 0 else 1.0

def hd95(prediction, target):
    if prediction.sum() == 0 or target.sum() == 0:
        return float("nan")
    target_distance = distance_transform_edt(1 - target)
    prediction_distance = distance_transform_edt(1 - prediction)
    distances = np.concatenate([
        target_distance[prediction > 0],
        prediction_distance[target > 0],
    ])
    return float(np.percentile(distances, 95))

def evaluate_episode_metrics(logits, target, threshold=0.5):
    prediction = binarize(logits, threshold)[0, 0].cpu().numpy()
    target_array = target[0, 0].cpu().numpy()
    iou, dice = overlap_metrics(prediction, target_array)
    return {
        "mIoU": iou,
        "Dice": dice,
        "BoundaryIoU": boundary_iou(prediction, target_array),
        "HD95": hd95(prediction, target_array),
    }
