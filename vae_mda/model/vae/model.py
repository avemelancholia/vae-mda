import torch
import os
import numpy as np

from torch import nn
from diffusers import AutoencoderKL
from diffusers.configuration_utils import FrozenDict

from .base_model import VAEModel


class DiagonalGaussianDistribution():
    def __init__(self, parameters: torch.Tensor, deterministic: bool=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn(
            self.mean.shape,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self) -> torch.Tensor:

        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1],
        )

    def mode(self) -> torch.Tensor:
        return self.mean

class RoundEmbeding(nn.Module):
    def __init__(
        self,
        emb_size=(39, 29),
    ):  
        super().__init__()
        self.emb_size = emb_size
        self.emb = nn.Linear(
            in_features=2,
            out_features=emb_size[0]*emb_size[1],
        )
        self.rad_per_day = 2*np.pi/366.0
    
    def forward(self, day):
        day = day.unsqueeze(dim=1)
        assert len(day.shape)==2
        assert day.shape[1]==1
        
        cos_val = torch.cos(self.rad_per_day*day)
        sin_val = torch.sin(self.rad_per_day*day)
        
        feat = torch.cat([cos_val, sin_val], dim=1)
        emb = self.emb(feat)

        return emb.reshape(-1, 1, self.emb_size[0], self.emb_size[1])
    

class CustomAutoencoderKL(nn.Module, VAEModel):
    def __init__(
        self,
        vae_config=None,
        basic_model_name=None, basic_model_subfolder=None,
        add_coord=False, linear=None, date_embedding=None,
        path=None
    ):  
        super().__init__()
        self.path = path
        self.linear = linear
        if self.linear:
            latent_channels = vae_config['latent_channels']
            self.encoder_linear = nn.Sequential(
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(
                    in_features=2*latent_channels*39*29,
                    out_features=2*linear
                ),
            )
            self.decoder_linear = nn.Sequential(
                nn.Linear(in_features=linear, out_features=latent_channels*39*29),
                nn.SiLU(),
                nn.Unflatten(1, (latent_channels, 39, 29)),
            )

        if add_coord:
            vae_config['in_channels'] += 2

        self.model = AutoencoderKL.from_config(vae_config)

        if date_embedding == 'round':
            self.conditional = True
            # для кодирования даты
            self.emb = RoundEmbeding()
            # куда подмешивать
            old_conv = self.model.post_quant_conv
            in_channels = old_conv.in_channels + 1
            block_out_channels = old_conv.out_channels
            self.model.post_quant_conv = nn.Conv2d(in_channels, block_out_channels, 1)
        else:
            self.conditional = False

    def _linear_lat(self, x):
        x = self.model.encoder(x)
        x = self.encoder_linear(x)
        lat = DiagonalGaussianDistribution(x)
        return lat
    
    def encode_sample(self, x):
        if self.linear:
            lat = self._linear_lat(x)
        else:
            lat = self.model.encode(x).latent_dist
        return lat.sample(), lat.kl().mean()
    
    def encode(self, x):
        if self.linear:
            lat = self._linear_lat(x)
        else:
            lat = self.model.encode(x).latent_dist
        return lat.mode(), lat.kl().mean()

    def decode(self, lat, t=None):
        if self.linear:
            lat = self.decoder_linear(lat)
        if self.conditional:
            cond = self.emb(t)
            lat = torch.cat([lat, cond], dim=1)
        return self.model.decode(lat).sample

    def save_pretrained(self, log_dir):
        if self.linear or self.conditional:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            torch.save(
                self.state_dict(),
                log_dir/"model.pt",
        )
        else:
            self.model.save_pretrained(log_dir)

    def from_pretrained(self, log_dir):
        if self.linear or self.conditional:
            checkpoint = torch.load(
                log_dir/'model.pt',
                weights_only=True,
                map_location="cpu",
            )
            self.load_state_dict(checkpoint)
        else:
            self.model = self.model.from_pretrained(log_dir)


def create_model(config):
    if 'basic_model_name' not in config:
        config['basic_model_name'] = None
    if 'basic_model_subfolder' not in config:
        config['basic_model_subfolder'] = None
    if 'add_coord' not in config:
        config['add_coord'] = False
    if 'linear' not in config:
        config['linear'] = None
    print(f"{config['linear']=}")
    if 'date_embedding' not in config:
        config['date_embedding'] = None
    model = CustomAutoencoderKL(
        vae_config=config['vae_config'],
        basic_model_name=config['basic_model_name'],
        basic_model_subfolder=config['basic_model_subfolder'],
        add_coord=config['add_coord'],
        linear=config['linear'],
        date_embedding=config['date_embedding'],
        path=config['path'],
    )
    return model
