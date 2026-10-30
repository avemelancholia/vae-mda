import torch
import numpy as np

from pathlib import Path

from .base_transform import BaseTransformer
from .sral_transform import sral_disc
from .utils import (
    fill_mask, nomalization, denomalization, init_siconc_quantile_mapper,
)


class SIVarTransformer(BaseTransformer):
    name = 'sea_ice_var'
    def __init__(self, config):
        self.indxs = config['indxs']
        self.means = np.array(
            config['means']
        )[:, np.newaxis, np.newaxis]
        
        self.stds = np.array(
            config['stds']
        )[:, np.newaxis, np.newaxis]
        self.padding = config['padding']
        
        self.mask = np.load(
            Path(config['mask_path'])
        )
        if len(self.indxs)==1:
            self.mask = self.mask[np.newaxis, :]
        else:
            self.mask = np.stack(
                [self.mask for _ in len(self.indxs)],
                axis=0,
            )
        self.dtype = torch.float
        if 'sral_indx' in config:
            print(f"{config['sral_indx']=}")
            self.sral_transfom = sral_disc[config['sral_indx']]['transform']
        else:
            self.sral_transfom = None
        self.siconc_quantile_mapper = init_siconc_quantile_mapper(config)

    def _map_observed_siconc(self, x):
        if self.siconc_quantile_mapper is None:
            return x
        return self.siconc_quantile_mapper(x)
    
    def back_transform(self, x):
        x = np.take(x, self.indxs, axis=0)
        if len(x.shape)==2:
            x = x[np.newaxis, :]
        x = fill_mask(x, self.mask, self.padding)
        return x

    def transform(self, x):
        # x [ch, w, h] numpy
        x = self.back_transform(x)
        x = nomalization(x, self.means, self.stds)
        # x [ch, w, h] torch
        return torch.tensor(x, dtype=self.dtype), torch.tensor(self.mask)

    def detransform(self, x):
        #[batch, ch, w, h] torch
        x = x.detach().cpu().numpy()
        means = self.means[None, ...]
        stds = self.stds[None, ...]
        x = denomalization(x, means, stds)
        x[:, 0] = np.clip(x[:, 0], 0, 1)
        #[batch, ch, w, h] numpy
        return x

    def val_transform(self, x):
        x = self._map_observed_siconc(x)
        x = x[None, :]
        x = fill_mask(x, self.mask)
        return torch.tensor(x, dtype=self.dtype)
    
    def val_sat_transform(self, y):
        # Преобразуем, как предписано
        if self.sral_transfom is not None:
            y = self.sral_transfom(y)
        y = self._map_observed_siconc(y)
        # Добавляем размерность
        y = y[None, ...]
        y = fill_mask(y, self.mask[None, 0, ...])
        return torch.tensor(y, dtype=self.dtype)

    def val_model_transform(self, x):
        x = self.back_transform(x)
        return torch.tensor(x, dtype=self.dtype)

    def obs_transform(self, y):
        # Преобразуем, как предписано
        if self.sral_transfom is not None:
            y = self.sral_transfom(y)
        y = self._map_observed_siconc(y)
        y = fill_mask(y, self.mask[0])
        y = nomalization(y, self.means[0], self.stds[0])
        return torch.tensor(y, dtype=self.dtype)

    def obs_model_transform(self, x, mask):
        # x [ch, w, h] numpy
        x = self.back_transform(x)
        y = np.zeros(x[0].shape)
        y.fill(np.nan)
        y[~mask] = x[0][~mask]
        y = nomalization(y, self.means[0], self.stds[0])
        return torch.tensor(y, dtype=self.dtype)

    def obs_detransform(self, y):
        #[batch, ch, w, h] torch
        y = y.detach().cpu().numpy()
        means = self.means[None, 0, ...]
        stds = self.stds[None, 0, ...]
        y = denomalization(y, means, stds)
        return y
