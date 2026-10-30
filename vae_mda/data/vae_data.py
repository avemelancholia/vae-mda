import torch
import numpy as np
import random
import pendulum as pdl

from pathlib import Path
from torch.utils.data import Dataset

from .transform.vae_transform import (
    VaeTransformer,
)
from .utils import(
    get_df,
)


transformers = {
    VaeTransformer.name: VaeTransformer,
}


class VAETrainDS(Dataset):
    name = 'vae_train_ds'

    def __init__(self, config, tag='train', time=None):
        preds_path = Path(config['dataset_dir']) / "preds"

        self.tag = tag
        
        self.train_start_day = np.datetime64(config['train_start_day'])
        self.train_end_day = np.datetime64(config['train_end_day'])
        self.val_start_day = np.datetime64(config['val_start_day'])
        self.val_end_day = np.datetime64(config['val_end_day'])
        
        self.transformer = transformers[config['transform_name']](config)

        data = get_df(preds_path)
        if tag == "train":
            start_day, end_day = self.train_start_day, self.train_end_day
        elif tag == "val":
            start_day, end_day = self.val_start_day, self.val_end_day
        else:
            raise ValueError(f"Unknown VAE dataset split: {tag!r}")
        self.data = data[(data.date >= start_day) & (data.date <= end_day)]
        if self.data.empty:
            raise ValueError(
                f"No {tag} predictions in {preds_path} for {start_day}..{end_day}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        day = pdl.datetime(
            row.date.year,
            row.date.month,
            row.date.day,
        )
        back_field = np.load(row.pred)
        i = random.randint(0, 23)
        field = back_field[:, i]
        x, mask = self.transformer.transform(field)
        return x, mask, day.day_of_year
