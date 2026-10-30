import torch
import numpy as np

from pathlib import Path
from torchvision import transforms 
from torchvision.transforms import v2

from .base_transform import BaseTransformer
from .sral_transform import sral_disc
from .utils import (
    nan_to_pad, apply_mask, fill_mask, nomalization, denomalization,
    init_siconc_quantile_mapper,
)

class VaeTransformer(BaseTransformer):
    name = "vae"
    def __init__(self, config): 
        self.indxs = config['indxs']

        if 'sral_indx' in config:
            print(f"{config['sral_indx']=}")
            self.sral_transfom = sral_disc[config['sral_indx']]['transform']
        else:
            self.sral_transfom = None
        self.sral_obs_transform = self.sral_transfom
        self.sral_val_sat_transform = self.sral_transfom
        if 'sral_obs_indx' in config:
            print(f"{config['sral_obs_indx']=}")
            self.sral_obs_transform = sral_disc[config['sral_obs_indx']]['transform']
        if 'sral_val_sat_indx' in config:
            print(f"{config['sral_val_sat_indx']=}")
            self.sral_val_sat_transform = sral_disc[config['sral_val_sat_indx']]['transform']
        self.sral_obs_confidence_transform = None
        if config.get('use_obs_confidence', False):
            confidence_index = config.get(
                'sral_obs_confidence_indx',
                config.get('sral_obs_indx', config.get('sral_indx')),
            )
            print(f"{confidence_index=}")
            confidence_entry = sral_disc[confidence_index]
            if 'confidence' not in confidence_entry:
                raise KeyError(
                    f"sral_obs_confidence_indx={confidence_index} has no confidence transform"
                )
            self.sral_obs_confidence_transform = confidence_entry['confidence']
        self.siconc_quantile_mapper = init_siconc_quantile_mapper(config)
        self.siconc_quantile_map_skip_validation = bool(
            config.get('siconc_quantile_map_skip_validation', False)
        )
        
        self.means = torch.tensor(
            config['means']
        )[..., None, None]
        self.stds = torch.tensor(
            config['stds']
        )[..., None, None]

        self.mask = np.load(
            Path(config['mask_path'])
        )
        self.add_gauss = config['add_gauss']
        self.gauss = v2.GaussianNoise(
            mean=0.0, sigma=0.001, clip=True,
        )

        if len(self.indxs)==1:
            self.mask = torch.tensor(self.mask[None, :])
        else:
            self.mask = torch.stack(
                [torch.tensor(self.mask) for _ in range(len(self.indxs))],
                dim=0,
            )
        if 'ice_mask_ind' not in config:
            self.ice_mask_ind = None
        else:
            self.ice_mask_ind = config['ice_mask_ind']
       
        self.dtype = torch.float
        self.padding = config["padding"]

        # Паддинг рамки для поля
        self.p = transforms.Pad((4, 1, 3, 0), fill=float('nan'))
        # Паддинг рамки для маски
        self.mp = transforms.Pad((4, 1, 3, 0), fill=True)
        # Перевод в тензор и паддинг рамки для поля
        self.tp = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(
                x, dtype=self.dtype
            )),
            self.p,
        ])
        # Замена Nan на self.padding
        self.n = transforms.Lambda(lambda x: nan_to_pad(x, self.padding))
        # Обрезка рамки
        self.dp = transforms.Lambda(lambda x: x[..., 1:, 4:-3])

    def _map_observed_siconc(self, x):
        if self.siconc_quantile_mapper is None:
            return x
        return self.siconc_quantile_mapper(x)

    def back_transform(self, x):
        """Получаем поля по индексам и добавляем размерность, если поле одно"""
        x = np.take(x, self.indxs, axis=0)
        if len(x.shape)==2:
            x = x[np.newaxis, :]
        return x

    def transform(self, x):
        # x numpy [ch, h, w]

        # Получаем поля по индексам и добавляем размерность, если поле одно
        x = self.back_transform(x)
        # Перевод в тензор и паддинг рамки для поля
        x = self.tp(x)

        # Маска
        true_mask = torch.isnan(x)
        # Паддинг рамки для маски
        mask = self.mp(self.mask.clone())
        # Если считаем маску по концентрации льда
        if self.ice_mask_ind is not None:
            for i in self.ice_mask_ind:
                mask[i] = true_mask[i]

        # Замена Nan на self.padding        
        x = self.n(x)
        if self.add_gauss:
            x[0] = self.gauss(x[0])
        # Нормализация
        x = nomalization(
            x, self.means.to(x.device), self.stds.to(x.device)
        )
        return x, mask

    def detransform(self, x, mask=None):
        # Режем рамку у поля
        x = self.dp(x)
        if mask is not None:
            # Режем рамку у маски 
            mask = self.dp(mask)
        else:
            mask = self.mask.clone()

        # Денормализация
        means = self.means[None, ...].to(x.device)
        stds = self.stds[None, ...].to(x.device)
        x = denomalization(x, means, stds)

        # Обрезаем поле льда
        x[:, 0] = torch.clamp(x[:, 0].cpu(), 0, 1)
        # Применяем маску
        x = apply_mask(x, mask)
        x = x.detach().cpu().numpy()
        return x

    def obs_detransform(self, y):
        #[batch, ch, w, h] torch
        y = self.dp(y)
        means = self.means[None, 0, ...].to(y.device)
        stds = self.stds[None, 0, ...].to(y.device)
        y = denomalization(y, means, stds)
        return y[:, 0].detach().cpu().numpy()

    # ------------------ усвоение со спутника ---------------
    def val_transform(self, x):
        if not self.siconc_quantile_map_skip_validation:
            x = self._map_observed_siconc(x)
        # Добавить размерность
        x = x[None, :]
        # x[mask] -> 'nan'
        mask = self.mask.clone()
        if x.shape[0]==1 and mask.shape[0]!=1:
            mask = mask[0]
            mask = mask[None, :]
        x = fill_mask(x, mask)
        return torch.tensor(x, dtype=self.dtype)

    def val_sat_transform(self, y):
        # Преобразуем, как предписано
        if self.sral_val_sat_transform is not None:
            y = self.sral_val_sat_transform(y)
        if not self.siconc_quantile_map_skip_validation:
            y = self._map_observed_siconc(y)
        # Добавляем размерность
        y = y[None, ...]
        y = fill_mask(y, self.mask[None, 0, ...].clone())
        return torch.tensor(y, dtype=self.dtype)

    def obs_transform(self, y):
        # Преобразуем, как предписано
        if self.sral_obs_transform is not None:
            y = self.sral_obs_transform(y)
        y = self._map_observed_siconc(y)
        # Паддим нанами сушу
        y = fill_mask(y, self.mask[0].clone().cpu().numpy())
        y = torch.tensor(y, dtype=self.dtype)
        # Нормализуем как поле
        y = nomalization(
            y, self.means[0].to(y.device), self.stds[0].to(y.device)
        )
        # Паддинг рамки для поля
        y = self.p(y)
        return y

    def obs_confidence_transform(self, y):
        if self.sral_obs_confidence_transform is None:
            raise ValueError("Observation confidence transform is not configured")
        y = self.sral_obs_confidence_transform(y)
        y = fill_mask(y, self.mask[0].clone().cpu().numpy())
        y = torch.tensor(y, dtype=self.dtype)
        y = self.p(y)
        return y

    # ------------------ усвоение с моедли ---------------
    def val_model_transform(self, x):
        # Получаем поля по индексам и добавляем размерность, если поле одно
        x = self.back_transform(x)
        x = torch.tensor(x, dtype=self.dtype)
        true_mask = torch.isnan(x)
        x = self.n(x)
        # Преобразуем маску
        mask = self.mask.clone()
        # Если считаем маску по концентрации льда
        if self.ice_mask_ind is not None:
            for i in self.ice_mask_ind:
                mask[i] = true_mask[i]
        x = apply_mask(x, mask)
        return x


    def obs_model_transform(self, x, mask):
        # x [ch, w, h] numpy
        # Получаем поля по индексам и добавляем размерность, если поле одно
        x = self.back_transform(x)
        # Перевод в тензор и паддинг рамки для поля
        x = self.tp(x)
        x = self.n(x)
        
        y = np.zeros(x[0].shape)
        y.fill(np.nan)
        y = torch.tensor(y, dtype=self.dtype, device=x.device)
        y[~mask] = x[0][~mask]
        # Нормализуем как поле
        y = nomalization(
            y, self.means[0].to(y.device), self.stds[0].to(y.device)
        )
        return y
