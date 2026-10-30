import json
import math
import pickle
import torch
import numpy as np


def fill_mask(x, mask, filler=float('nan')):
    print(f"{x.shape=} {mask.shape=}")
    if isinstance(filler, list):
        for i in range(len(filler)):
            mask_i = np.isnan(x[i])
            x[i][np.logical_and(~mask[i], mask_i)] = filler[i]
    x[mask] = float('nan')
    return x

def nan_to_pad(x, padding):
    for i in range(len(padding)):
        x[i] = torch.nan_to_num(
            x[i], nan=padding[i])
    return x

def apply_mask(x, mask):
    # torch[batch, ch, w, h]
    if len(x.shape)!=len(mask.shape):
        mask = torch.stack([
            mask for _ in range(x.shape[0])
        ])
    x[mask] = float('nan')
    return x

def nomalization(x, means, stds):
    # x[ch, w, h], means[ch, 1, 1], stds[ch, 1, 1]
    return (x - means)/stds

def denomalization(x, means, stds):
    # x[ch, w, h], means[ch, 1, 1], stds[ch, 1, 1]
    return x*stds + means

class QuantileTransform():
    def __init__(self, json_dir):
        with open(json_dir, 'r') as j:
            self.rule = json.loads(j.read())
        self.keys = list(self.rule.keys())
        self.sort_idx = np.argsort(self.keys)
    
    def __call__(self, sral: np.array):
        shape = sral.shape
        mask = np.isnan(sral)
        vals = sral[~mask]

        idx = np.searchsorted(self.keys, vals, sorter=self.sort_idx)
        vals = np.asarray(self.keys)[self.sort_idx][idx]
        
        sral = np.zeros(shape)
        sral.fill(np.nan)
        sral[~mask] = vals
        return sral


class EmpiricalQuantileMapper:
    def __init__(self, path, direction="amsr2_to_model", preserve_bounds=True):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        if "directions" not in payload:
            raise KeyError(f"Quantile map {path} does not contain 'directions'")
        if direction not in payload["directions"]:
            available = sorted(payload["directions"])
            raise KeyError(
                f"Direction {direction!r} not found in {path}. Available: {available}"
            )

        mapping = payload["directions"][direction]
        metadata = payload.get("metadata", {})
        source = np.asarray(mapping["source_quantiles"], dtype=np.float64)
        target = np.asarray(mapping["target_quantiles"], dtype=np.float64)
        finite = np.isfinite(source) & np.isfinite(target)
        source, target = source[finite], target[finite]

        order = np.argsort(source)
        source, target = source[order], target[order]
        source, unique_idx = np.unique(source, return_index=True)
        target = target[unique_idx]

        if len(source) < 2:
            raise ValueError(f"Quantile map {path} direction {direction!r} is degenerate")

        self.source_quantiles = source
        self.target_quantiles = target
        self.direction = direction
        self.preserve_bounds = preserve_bounds
        self.min_value = metadata.get("min_value")
        self.max_value = metadata.get("max_value")

    def __call__(self, x):
        x = np.asarray(x)
        out = np.full(x.shape, np.nan, dtype=np.float32)
        mask = np.isfinite(x)
        if not np.any(mask):
            return out

        vals = np.clip(x[mask].astype(np.float64), 0.0, 1.0)
        mapped = np.interp(
            vals,
            self.source_quantiles,
            self.target_quantiles,
            left=self.target_quantiles[0],
            right=self.target_quantiles[-1],
        )
        mapped = np.clip(mapped, 0.0, 1.0)
        if self.preserve_bounds:
            mapped[vals <= 0.0] = 0.0
            mapped[vals >= 1.0] = 1.0
        if self.min_value is not None:
            mapped[vals < self.min_value] = vals[vals < self.min_value]
        if self.max_value is not None:
            mapped[vals > self.max_value] = vals[vals > self.max_value]
        out[mask] = mapped.astype(np.float32)
        return out


def init_siconc_quantile_mapper(config):
    path = config.get("siconc_quantile_map_path")
    if not path:
        return None
    direction = config.get("siconc_quantile_map_direction", "amsr2_to_model")
    preserve_bounds = config.get("siconc_quantile_map_preserve_bounds", True)
    print(f"Using siconc quantile map: {path} ({direction})")
    return EmpiricalQuantileMapper(path, direction, preserve_bounds)
