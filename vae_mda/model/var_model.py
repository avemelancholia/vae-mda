import torch
import torch.nn as nn
import numpy as np
import copy

from .base_model import AssimilationModel
from .var.polynomial_matrix import PolynomialMatrix

   
class Loss3DVar(nn.Module):
    def __init__(self, inv_background_matrix, inv_obs_matrix_sigma2):
        super().__init__()
        self.inv_b = inv_background_matrix
        self.inv_r_sigma2 = inv_obs_matrix_sigma2
        self.losses = []


    def forward(self, x, x_background, y, y_background):
        """Formula [source] (https://www.mdpi.com/2077-1312/10/3/432)"""
        self.inv_r = torch.eye(y.numel(), dtype=torch.float)/self.inv_r_sigma2
        delta_x = (x - x_background).view([1, x.shape[0]])
        delta_y = (y - y_background).view([1, y.shape[0]]).to(delta_x.dtype)

        loss_background = delta_x @ self.inv_b @ torch.transpose(delta_x, 0, 1)
        loss_observation = delta_y @ self.inv_r @ torch.transpose(delta_y, 0, 1)

        loss = loss_background + loss_observation
        self.losses.append(loss.detach().cpu().numpy().item())
        return loss


class Loss3DVarEXP(nn.Module):
    def __init__(self, inv_background_matrix, inv_obs_matrix_sigma2, depth, factor):
        super().__init__()
        self.inv_b = inv_background_matrix
        self.inv_r_sigma2 = inv_obs_matrix_sigma2
        self.depth = depth
        self.factor = factor
        self.losses = []

    def forward(self, x, x_background, y, y_background):
        """Formula [source] (https://www.mdpi.com/2077-1312/10/3/432)"""
        delta_x = (x - x_background).view([1, x.shape[0]])
        loss_background = delta_x @ self.inv_b @ torch.transpose(delta_x, 0, 1)
        loss = 0
        for i in y_background:
            self.inv_r = torch.eye(y[i].numel(), dtype=torch.float)/self.inv_r_sigma2
            delta_y = (y[i] - y_background[i]).view([1, y[i].shape[0]]).to(delta_x.dtype)
            loss += (self.factor**i) * (delta_y @ self.inv_r @ torch.transpose(delta_y, 0, 1) + loss_background)

        self.losses.append(loss.detach().cpu().numpy().item())
        return loss


class Model3Dvar(AssimilationModel):
    name = "3d_var_baseline"

    def __init__(self, inv_corr_matrix, inv_obs_matrix_sigma2, metric_mask):
        super().__init__()
        self.inv_corr_matrix = inv_corr_matrix
        self.loss = Loss3DVar(inv_corr_matrix, inv_obs_matrix_sigma2)
        self.losses = []
        self.make_val = True
        self.metric_mask = metric_mask

    @staticmethod
    def init_params(config, full_config=None):
        if config["load_inv_corr_matrix"]:
            inv_matrix = torch.tensor(
                np.load(config["inv_corr_matrix_path"]), 
                dtype=torch.float
            )
        elif config["corr_matrix_tag"]=="polynomial":
            inv_matrix = PolynomialMatrix(config).inv_matrix
        config["load_inv_corr_matrix"] = True

        
        return {
            "inv_corr_matrix":
                inv_matrix / config["inv_corr_matrix_sigma2"],
            "inv_obs_matrix_sigma2": config["inv_obs_matrix_sigma2"],
            "metric_mask": np.load(config["mask_path"])
        }

    def preprocess_tensors(self, tensors, config):
        x, x_b, y_b = tensors
        return [
            x, 
            x_b,
            y_b
        ]

    def fit(self, tensors, steps, meta=None):
        def closure():
            optimizer.zero_grad()
            y_b = self.y_init.clone().detach()
            x_b = self.x_b_init.clone().detach()
            mask_x, mask_y = self.get_valid_mask(x_b, y_b)
            x = self.x[mask_x].flatten()
            x_b = x_b[mask_x].flatten()
            y = self.x[mask_y].flatten()
            y_b = y_b[mask_y].flatten()
            loss = self.loss(x, x_b, y, y_b)
            loss.backward()
            return loss

        x_init, x_b_init, y_init = tensors
        self.x_b_init = x_b_init[0, 0].clone().detach()
        self.y_init = y_init[0].clone().detach()
        self.x = x_init[0, 0].clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.x])

        for n in range(steps):
            optimizer.step(closure)

        self.losses = self.loss.losses.copy()
        self.loss.losses = []


    @staticmethod
    def get_valid_mask(x, y):
        x_mask = ~x.isnan()
        y_mask = x_mask & (~y.isnan())
        return x_mask, y_mask

    def get_x(self):
        return self.x_b_init[None, None, ...], self.x[None, None, ...]

    def get_obs(self):
        return self.y_init[None, ...]


class Model3DvarEXP(Model3Dvar):
    name = "3d_var_exp"

    def __init__(self, inv_corr_matrix, inv_obs_matrix_sigma2, metric_mask, depth, factor):
        super().__init__(inv_corr_matrix, inv_obs_matrix_sigma2, metric_mask)
        self.loss = Loss3DVarEXP(inv_corr_matrix, inv_obs_matrix_sigma2, depth, factor)

    @staticmethod
    def init_params(config, full_config=None):
        if config["load_inv_corr_matrix"]:
            inv_matrix = torch.tensor(
                np.load(config["inv_corr_matrix_path"]), 
                dtype=torch.float
            )
        elif config["corr_matrix_tag"]=="polynomial":
            inv_matrix = PolynomialMatrix(config).inv_matrix
        config["load_inv_corr_matrix"] = True

        return {
            "inv_corr_matrix":
                inv_matrix / config["inv_corr_matrix_sigma2"],
            "inv_obs_matrix_sigma2": config["inv_obs_matrix_sigma2"],
            "metric_mask": np.load(config["mask_path"]),
            "depth": full_config["depth"],
            "factor": full_config["factor"],
        }

    def preprocess_tensors(self, tensors, config):
        x, x_b, y_b = tensors
        return [x, x_b, y_b]

    def fit(self, tensors, steps, meta=None):
        def closure():
            optimizer.zero_grad()
            y_b = copy.deepcopy(self.y_init)
            x_b = self.x_b_init.clone().detach()
            
            y = {}
            for i in y_b:
                mask_x, mask_y = self.get_valid_mask(x_b, y_b[i])
                y[i] = self.x[mask_y].flatten()
                y_b[i] = y_b[i][mask_y].flatten()
            
            x = self.x[mask_x].flatten()
            x_b = x_b[mask_x].flatten()
            loss = self.loss(x, x_b, y, y_b)
            loss.backward()
            return loss
        
        x_init, x_b_init, y_init = tensors
        self.x_b_init = x_b_init[0, 0].clone().detach()
        self.y_init = {key: y_init[key][0] for key in y_init}
        self.x = x_init[0, 0].clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.x])

        for n in range(steps):
            optimizer.step(closure)

        self.losses = self.loss.losses.copy()
        self.x = torch.clip(self.x, min=0, max=1)
        self.loss.losses = []

    def get_obs(self):
        return self.y_init[0][None, ...]
