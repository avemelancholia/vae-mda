import os
import numpy as np
import pandas as pd
import pendulum as pdl
import torch
import random
import json

from pathlib import Path
from torchvision import transforms 
from torch.utils.data.dataset import Dataset
from datetime import datetime, timedelta

from .utils import (
    get_df,
    nemo_folder,
    get_nc_data,
)
from .transform.var_transform import SIVarTransformer
from .transform.vae_transform import VaeTransformer

transformers = {
    SIVarTransformer.name: SIVarTransformer,
    VaeTransformer.name: VaeTransformer,
}

class InferenceDS(Dataset):
    name = "InferenceDS"
    def __init__(self, config, data_config):
        self.date = pdl.parse(config['date'])
        self.background = get_nc_data(
            config['restart_dir'], config['wrf_dir'], self.date
        )[:, 23]
        self.transformer = transformers[data_config['transform_name']](data_config)
        sral_dir = Path(config['sral_dir'])
        print(f"{sral_dir=}")
        self.sral_files = []
        curr_date =  pdl.parse(config['date'])
        print(f"{data_config['assimilation_range']=}")
        for i in range(data_config['assimilation_range']):
            self.sral_files.append(sral_dir/f"{curr_date.to_date_string()}.npy")
            curr_date = curr_date - pdl.duration(days=1)
    
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        back_field, back_mask = self.transformer.transform(self.background)
        
        obs_field = np.zeros((1, 1, 1))
        for dir_ in self.sral_files:
            try: 
                sral_field = np.load(dir_)
            except FileNotFoundError:
                print(f"File {dir_} not found")
                continue
            if obs_field.shape!=sral_field.shape:
                obs_field = np.zeros(sral_field.shape)
                obs_field.fill(np.nan)
            obs_field[np.isnan(obs_field)] = sral_field[np.isnan(obs_field)]
        obs_field = self.transformer.obs_transform(obs_field)
        
        return (
            (back_field, back_field, obs_field),
            {
                "date": self.date.to_date_string(),
                "back_mask": back_mask,
                "day_of_year": self.date.day_of_year,
            },
        )
