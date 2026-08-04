import torch
import functools

import numpy as np


def init_metrics(mask_land, sparse_gt=False):
    metrics = []

    def extract_values(func):
        @functools.wraps(func)
        def fix_values(pred, gt):
            pred = pred.copy()
            gt = gt.copy()

            mask_ocean = ~mask_land
            nans_pred = np.isnan(pred)
            nans_gt = np.isnan(gt)

            if sparse_gt:
                correct_mask = mask_ocean & (~nans_gt) & (~np.isnan(pred))
            else:
                pred[nans_pred & mask_ocean] = 0.0
                gt[nans_gt & mask_ocean] = 0.0
                assert np.all(np.isnan(pred) == np.isnan(gt))
                correct_mask = ~np.isnan(pred)

            return func(pred[correct_mask], gt[correct_mask])

        return fix_values

    @extract_values
    def mae(pred, gt):
        return np.mean(np.abs(gt - pred))

    metrics.append(mae)

    @extract_values
    def iiee(pred, gt):
        threshold = 0.15
        pred_t = pred > threshold
        gt_t = gt > threshold
        not_equal = pred_t != gt_t
        return np.mean(not_equal)

    metrics.append(iiee)

    @extract_values
    def rmse(pred, gt):
        return np.sqrt(np.mean((gt - pred) ** 2))

    metrics.append(rmse)

    @extract_values
    def mse(pred, gt):
        return np.mean((gt - pred) ** 2)

    metrics.append(mse)
    return metrics
