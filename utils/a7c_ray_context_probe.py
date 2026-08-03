from __future__ import annotations

import numpy as np
import torch


def exact_alpha_transmittance_mass(rendered, attribution_colors, *, retain_graph=False):
    if rendered.ndim != 3 or attribution_colors.ndim != 2:
        raise ValueError("rendered and attribution colors have invalid shapes")
    gradient = torch.autograd.grad(
        rendered.sum(), attribution_colors, retain_graph=retain_graph, create_graph=False
    )[0]
    return gradient.mean(dim=1)


def ray_depth_moments(alpha, depth_numerator, depth2_numerator, epsilon=1.0e-8):
    a = np.asarray(alpha, dtype=np.float64)
    d = np.asarray(depth_numerator, dtype=np.float64)
    d2 = np.asarray(depth2_numerator, dtype=np.float64)
    if a.shape != d.shape or a.shape != d2.shape:
        raise ValueError("ray buffers must have matching shapes")
    available = a > float(epsilon)
    mean = np.where(available, d / np.maximum(a, epsilon), 0.0)
    variance = np.where(available, np.maximum(d2 / np.maximum(a, epsilon) - mean * mean, 0.0), 0.0)
    return mean, variance, available.astype(np.float32)


def sample_footprint_context(buffers, *, projected_xy, radii):
    values = np.asarray(buffers, dtype=np.float64)
    if values.ndim == 2:
        values = values[None]
    xy = np.asarray(projected_xy, dtype=np.float64)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or xy.shape != (radius_values.size, 2):
        raise ValueError("buffer or projected carrier shapes are invalid")
    channels, height, width = values.shape
    output = np.zeros((radius_values.size, channels * 3), dtype=np.float32)
    for index, ((x, y), radius) in enumerate(zip(xy, radius_values)):
        cx = int(np.clip(round(x), 0, width - 1))
        cy = int(np.clip(round(y), 0, height - 1))
        r = max(int(np.ceil(max(radius, 0.0))), 1)
        x0, x1 = max(cx - r, 0), min(cx + r + 1, width)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, height)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        patch = values[:, y0:y1, x0:x1][:, mask]
        center = values[:, cy, cx]
        if patch.shape[1] == 0:
            mean = center
            variance = np.zeros_like(center)
        else:
            mean = patch.mean(axis=1)
            variance = patch.var(axis=1)
        output[index] = np.concatenate([center, mean, variance])
    return output
