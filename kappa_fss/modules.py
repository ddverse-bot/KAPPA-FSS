import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_line_kernels(n_angles, length, width):
    kernels = torch.zeros(n_angles, 1, length, length)
    center = length // 2
    for angle_index in range(n_angles):
        theta = math.pi * angle_index / n_angles
        for position in np.linspace(-(length - 1) / 2, (length - 1) / 2, max(4 * length, 16)):
            for offset in np.linspace(-width / 2, width / 2, max(1, width)):
                x = int(round(center + position * math.cos(theta) - offset * math.sin(theta)))
                y = int(round(center + position * math.sin(theta) + offset * math.cos(theta)))
                if 0 <= x < length and 0 <= y < length:
                    kernels[angle_index, 0, y, x] = 1.0
        total = kernels[angle_index].sum()
        if total > 0:
            kernels[angle_index] /= total
    return kernels

class KTP(nn.Module):
    def __init__(self, in_channels, reduced=256, n_angles=12, lengths=(7, 11, 15), width=1, tau_init=0.1):
        super().__init__()
        self.n_angles = n_angles
        self.lengths = lengths
        self.reduce = nn.Conv2d(in_channels, reduced, 1)
        self.project = nn.Conv2d(reduced + 1, in_channels, 1)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        for length in lengths:
            self.register_buffer(f"kernels_{length}", make_line_kernels(n_angles, length, width))

    def _scale_forward(self, features, length):
        kernels = getattr(self, f"kernels_{length}")
        batch, channels, height, width = features.shape
        weights = kernels.repeat(channels, 1, 1, 1)
        expanded = features.abs().repeat_interleave(self.n_angles, dim=1)
        responses = F.conv2d(expanded, weights, padding=length // 2, groups=channels * self.n_angles)
        responses = responses.view(batch, channels, self.n_angles, height, width)
        temperature = self.log_tau.exp()
        attention = torch.softmax(responses / temperature, dim=2)
        pooled = (attention * responses).sum(2)
        entropy = -(attention * (attention + 1e-8).log()).sum(2)
        coherence = 1 - entropy.mean(1, keepdim=True) / math.log(self.n_angles)
        return pooled, coherence

    def forward(self, features):
        reduced = self.reduce(features)
        pooled_scales = []
        coherence_scales = []
        for length in self.lengths:
            pooled, coherence = self._scale_forward(reduced, length)
            pooled_scales.append(pooled)
            coherence_scales.append(coherence)
        pooled = torch.stack(pooled_scales, dim=0).sum(0)
        coherence = torch.stack(coherence_scales, dim=0).mean(0)
        residual = self.project(torch.cat([pooled, coherence], dim=1))
        return features + self.gamma * residual, coherence

def erode(mask, iterations):
    result = mask
    for _ in range(iterations):
        result = -F.max_pool2d(-result, 3, 1, 1)
    return result

def assouad_map(mask, small_scales=(1, 2, 4), large_scales=(16, 32, 64), temperature=0.1):
    boundary = mask - erode(mask, 1)
    slopes = []
    for large_scale in large_scales:
        log_counts = []
        log_ratios = []
        for small_scale in small_scales:
            occupancy = F.max_pool2d(boundary, small_scale, small_scale) if small_scale > 1 else boundary
            window = max(1, large_scale // small_scale)
            counts = F.avg_pool2d(occupancy, window, 1, window // 2) * (window ** 2)
            counts = F.interpolate(counts, size=mask.shape[-2:], mode="nearest")
            log_counts.append((counts + 1).log())
            log_ratios.append(math.log(large_scale / small_scale))
        x = torch.tensor(log_ratios, device=mask.device)
        centered_x = x - x.mean()
        y = torch.stack(log_counts, dim=0)
        slope = ((y - y.mean(0)) * centered_x.view(-1, 1, 1, 1, 1)).sum(0) / (centered_x ** 2).sum()
        slopes.append(slope)
    complexity = temperature * torch.logsumexp(torch.stack(slopes, dim=0) / temperature, dim=0)
    return complexity * (boundary > 0.05).float()

def support_region_masks(mask, boundary_width):
    interior = (erode(mask, boundary_width) > 0.5).float()
    foreground = (mask > 0.5).float()
    boundary = (foreground - interior).clamp(0, 1)
    background = 1.0 - foreground
    complexity = assouad_map(mask)
    values = complexity[0, 0][boundary[0, 0] > 0.5]
    median = values.median() if values.numel() else torch.tensor(0.0, device=mask.device)
    smooth = boundary * (complexity < median).float()
    complex_boundary = boundary * (complexity >= median).float()
    if smooth.sum() < 1:
        smooth = boundary.clone()
    if complex_boundary.sum() < 1:
        complex_boundary = boundary.clone()
    return {"interior": interior[0], "smooth": smooth[0], "complex": complex_boundary[0], "background": background[0]}

def build_prototypes(support_features, support_masks, boundary_width=8, feature_stride=16):
    channels, height, width = support_features[0].shape
    feature_boundary_width = max(1, boundary_width // feature_stride)
    sums = {key: [] for key in ("interior", "smooth", "complex", "background")}
    counts = {key: [] for key in sums}
    for features, mask in zip(support_features, support_masks):
        small_mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
        regions = support_region_masks(small_mask, feature_boundary_width)
        for key, region in regions.items():
            expanded = region.expand(channels, -1, -1)
            sums[key].append((features * expanded).sum(dim=(1, 2)))
            counts[key].append(region.sum())
    prototypes = {}
    for key in sums:
        numerator = torch.stack(sums[key], dim=0).sum(0)
        denominator = torch.stack(counts[key], dim=0).sum(0).clamp(min=1.0)
        prototypes[key] = numerator / denominator
    return prototypes

def cosine_prior_maps(query_features, prototypes):
    channels, height, width = query_features.shape
    normalized_query = F.normalize(query_features.view(channels, -1), dim=0)
    maps = []
    for key in ("interior", "smooth", "complex", "background"):
        prototype = F.normalize(prototypes[key].view(channels, 1), dim=0)
        maps.append((normalized_query * prototype).sum(0).view(1, height, width))
    return torch.cat(maps, dim=0)
