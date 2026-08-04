import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from diffusers import AutoencoderKL
from torchvision import transforms

from .base_model import AssimilationModel
from .vae.model import create_model as create_model_diff


create_model = {
    "diffusers": create_model_diff,
}

class LossVAE3DVar(nn.Module):
    def __init__(self, lat_weight, obs_weight, mod_weight):
        super().__init__()
        self.lat_w = lat_weight
        self.obs_w = obs_weight
        self.mod_w = mod_weight
        self.lat_loss = nn.MSELoss()
        self.obs_loss = nn.MSELoss()
        self.mod_loss = nn.MSELoss()

    def _obs_mse(self, pred, target, confidence=None):
        if confidence is None:
            return self.obs_loss(pred, target)
        confidence = torch.nan_to_num(confidence, nan=0.0)
        confidence_sum = torch.clamp(confidence.sum(), min=1e-12)
        return torch.sum(confidence * (pred - target) ** 2) / confidence_sum

    def forward(
        self, lat_pred, lat_init, mod_pred,
        mod_init, obs_pred, obs_init, obs_confidence=None
    ):
        loss = self.lat_w*self.lat_loss(lat_pred, lat_init) \
             + self.obs_w*self._obs_mse(obs_pred, obs_init, obs_confidence) \
             + self.mod_w*self.mod_loss(mod_pred, mod_init)
        return loss


class LossVAE3DVarEXP(nn.Module):
    def __init__(
        self, depth, factor, lat_weight=1.0,
        obs_weight=1.0, mod_weight=1.0
    ):
        super().__init__()
        self.lat_w = lat_weight
        self.mod_w = mod_weight
        self.obs_w = obs_weight

        self.depth = depth
        self.factor = factor
        self.losses = []

        self.lat_loss = nn.MSELoss()
        self.obs_loss = nn.MSELoss()
        self.mod_loss = nn.MSELoss()


    def forward(
        self, lat_pred, lat_init, mod_pred,
        mod_init, obs_preds, obs_inits
    ):
        """Formula [source] (https://www.mdpi.com/2077-1312/10/3/432)"""
        loss = self.lat_w*self.lat_loss(lat_pred, lat_init) \
            + self.mod_w*self.mod_loss(mod_pred, mod_init)
        for key in obs_preds:
            loss += self.obs_w*(self.factor**key) \
            * self.obs_loss(obs_preds[key], obs_inits[key])
            #self.losses.append(loss.detach().cpu().numpy().item())
        return loss


def _init_early_stopping(config, steps):
    if not config.get("early_stopping", False):
        return None

    return {
        "best_loss": float("inf"),
        "bad_steps": 0,
        "warmup": int(config.get("early_stopping_warmup", 0)),
        "patience": int(config.get("early_stopping_patience", 100)),
        "min_delta": float(config.get("early_stopping_min_delta", 0.0)),
    }


def _should_stop_early(state, loss_value, step):
    if state is None or step < state["warmup"]:
        return False

    if loss_value < state["best_loss"] - state["min_delta"]:
        state["best_loss"] = loss_value
        state["bad_steps"] = 0
        return False

    state["bad_steps"] += 1
    return state["bad_steps"] >= state["patience"]


def _make_latent_optimizer(params, optimizer_name, lr, amsgrad=False):
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, amsgrad=amsgrad)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, amsgrad=amsgrad)
    if optimizer_name == "lion":
        from lion_pytorch import Lion
        return Lion(params, lr=lr)
    raise ValueError(f"Unknown assimilation optimizer: {optimizer_name}")


def _make_latent_scheduler(optimizer, scheduler_name, steps, min_lr=0.0):
    if scheduler_name is None:
        return None
    scheduler_name = str(scheduler_name).lower()
    if scheduler_name in ("none", "null"):
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(steps), 1),
            eta_min=float(min_lr),
        )
    raise ValueError(f"Unknown assimilation scheduler: {scheduler_name}")


class VAEModel3Dvar(AssimilationModel):
    name = "vae_3d_var"

    @staticmethod
    def _move_model_to_device(model, device):
        model = model.to(device)
        if hasattr(model, 'model'):
            model.model = model.model.to(device)
        return model

    def __init__(
            self,
            model,
            lat_weight,
            obs_weight,
            mod_weight,
            add_sigmoid=False,
            assim_lr=0.001,
            assim_optimizer="adam",
            assim_scheduler=None,
            assim_min_lr=0.0,
            grad_clip_norm=None,
            amsgrad=False,
            background_loss_mode="siconc",
        ):
        super().__init__()
        self.model = model
        self.lr = assim_lr
        self.assim_optimizer = assim_optimizer
        self.assim_scheduler = assim_scheduler
        self.assim_min_lr = assim_min_lr
        self.grad_clip_norm = grad_clip_norm
        self.amsgrad = amsgrad
        self.loss = LossVAE3DVar(lat_weight, obs_weight, mod_weight)
        valid_background_loss_modes = {"siconc", "multivariate"}
        if background_loss_mode not in valid_background_loss_modes:
            raise ValueError(
                "background_loss_mode must be one of "
                f"{sorted(valid_background_loss_modes)}, got "
                f"{background_loss_mode!r}"
            )
        self.background_loss_mode = background_loss_mode
        self.add_sigmoid = add_sigmoid

    @staticmethod
    def init_params(config, full_config=None):
        path = Path(config['path'])
        device = full_config['device']
        creater = create_model[full_config['vae_type']]
        if 'ice_mask_inx' not in config:
            config['ice_mask_inx'] = None
        model = creater(config)
        model.from_pretrained(path)
        if 'add_sigmoid' not in config:
            config['add_sigmoid'] = False
        return {
            'model': VAEModel3Dvar._move_model_to_device(model, device),
            'lat_weight': full_config['lat_weight'],
            'obs_weight': full_config['obs_weight'],
            'mod_weight': full_config['mod_weight'],
            'background_loss_mode': full_config.get(
                'background_loss_mode', 'siconc'
            ),
            'add_sigmoid': config['add_sigmoid'],
            'assim_lr': full_config.get('assim_lr', 0.001),
            'assim_optimizer': full_config.get('assim_optimizer', 'adam'),
            'assim_scheduler': full_config.get('assim_scheduler'),
            'assim_min_lr': full_config.get('assim_min_lr', 0.0),
            'grad_clip_norm': full_config.get('grad_clip_norm'),
            'amsgrad': full_config.get('amsgrad', False),
        }

    @staticmethod
    def preprocess_tensors(tensors, config):
        if len(tensors) == 3:
            x, x_b, y_b = tensors
            return [x, x_b, y_b]
        if len(tensors) == 4:
            x, x_b, y_b, y_confidence = tensors
            return [x, x_b, y_b, y_confidence]
        raise ValueError(f"Expected 3 or 4 tensors, got {len(tensors)}")

    def _background_loss_inputs(self, x_pred, x_init, siconc_mask):
        if self.background_loss_mode == "siconc":
            return x_pred[0][siconc_mask], x_init[0][siconc_mask]

        finite_background = torch.isfinite(x_init)
        return x_pred[finite_background], x_init[finite_background]

    def fit(self, tensors, steps, meta=None):
        t = meta['day_of_year']
        config = meta.get("config", {}) if meta else {}
        validation_callback = meta.get("validation_callback") if meta else None
        track_validation = bool(
            config.get("track_validation_during_fit", False)
            and validation_callback is not None
        )
        validation_interval = int(config.get("validation_interval", 25))
        restore_best_validation_step = bool(
            config.get("restore_best_validation_step", False)
        )
        lat_zero_prior = bool(config.get("lat_zero_prior", False))
        best_validation_metric = config.get("best_validation_metric", "val_siconc_mae")
        fit_curve_metrics = set(config.get(
            "fit_curve_metrics",
            ["assimilation_loss", "val_siconc_mae", "val_siconc_rmse"],
        ))
        self.losses = {"assimilation_loss": []} if track_validation else []
        self.best_validation_step = None
        early_stopping = _init_early_stopping(
            config, steps
        )
        self.stopped_step = steps if early_stopping is not None else None
        if len(tensors) == 3:
            x_init, x_b_init, y_init = tensors
            y_confidence = None
        else:
            x_init, x_b_init, y_init, y_confidence_init = tensors
            y_confidence = y_confidence_init.clone().detach()
        self.y_shape = y_init.shape

        # init state
        y_obs = y_init.clone().detach()
        self.x_b_init = x_b_init.clone().detach()
        self.y_init = y_init.clone().detach()

        # save mask
        mask_x, mask_y = self.get_valid_mask(x_b_init[:, 0], y_init)

        # init lattent
        lat_init, _ = self.model.encode(x_b_init)
        if hasattr(self.model, 'prepare_for_assimilation'):
            self.model.prepare_for_assimilation()
        self.lat = lat_init.clone().detach()
        lat_target = torch.zeros_like(lat_init) if lat_zero_prior else lat_init

        self.lat.requires_grad_(True)
        optimizer = _make_latent_optimizer(
            [self.lat], self.assim_optimizer, self.lr, self.amsgrad
        )
        scheduler = _make_latent_scheduler(
            optimizer, self.assim_scheduler, steps, self.assim_min_lr
        )
        best_validation_value = float("inf")
        best_validation_lat = None

        for step in range(1, steps + 1):
            optimizer.zero_grad()
            x_pred = self.model.decode(self.lat, t)
            if self.add_sigmoid:
                x_pred[:, 0] = torch.sigmoid(x_pred[:, 0])
            loss = 0
            batch_size = x_pred.shape[0]
            for i in range(batch_size):
                mod_pred, mod_init = self._background_loss_inputs(
                    x_pred[i], x_init[i], mask_x[i]
                )
                loss += self.loss(
                    self.lat[i], lat_target[i].detach(),
                    mod_pred, mod_init.detach(),
                    x_pred[i, 0][mask_y[i]], y_obs[i][mask_y[i]].detach(),
                    None if y_confidence is None else y_confidence[i][mask_y[i]].detach(),
                )
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([self.lat], self.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = loss.detach().cpu().numpy().item()
            if track_validation:
                self.losses["assimilation_loss"].append((step, loss_value))
            else:
                self.losses.append(loss_value)

            if track_validation and (step == 1 or step % validation_interval == 0):
                with torch.no_grad():
                    x_val = self.model.decode(self.lat, t)
                    if self.add_sigmoid:
                        x_val[:, 0] = torch.sigmoid(x_val[:, 0])
                validation_metrics = validation_callback(x_val)
                for metric_name, metric_value in validation_metrics.items():
                    if metric_name in fit_curve_metrics:
                        self.losses.setdefault(metric_name, []).append((step, metric_value))
                metric_value = validation_metrics.get(best_validation_metric)
                if metric_value is not None and not np.isnan(metric_value):
                    if metric_value < best_validation_value:
                        best_validation_value = metric_value
                        best_validation_lat = self.lat.detach().clone()
                        self.best_validation_step = step

            if _should_stop_early(early_stopping, loss_value, step):
                self.stopped_step = step
                print(f"Early stopping at step {step}/{steps}")
                break

        if early_stopping is not None:
            self.stopped_step = min(self.stopped_step, steps)

        if restore_best_validation_step and best_validation_lat is not None:
            self.lat = best_validation_lat
            print(
                f"Restored best validation step {self.best_validation_step} "
                f"with {best_validation_metric}={best_validation_value:.6g}"
            )

        self.x = self.model.decode(self.lat, t)

        if self.add_sigmoid:
            self.x[:, 0] = torch.sigmoid(self.x[:, 0])

    @staticmethod
    def get_valid_mask(x, y):
        x_mask = ~x.isnan()
        y_mask = x_mask & (~y.isnan())
        return x_mask, y_mask

    def get_x(self):
        return self.x_b_init, self.x

    def get_obs(self):
        return self.y_init[None, ...]


class VAEModel3DvarEXP(VAEModel3Dvar):
    name = "vae_3d_var_exp"

    def __init__(
        self,
        model,
        depth,
        factor,
        lat_weight,
        obs_weight,
        mod_weight,
        add_sigmoid,
        assim_lr=0.001,
        assim_optimizer="adam",
        assim_scheduler=None,
        assim_min_lr=0.0,
        grad_clip_norm=None,
        amsgrad=False,
    ):
        super().__init__(
            model,
            lat_weight,
            obs_weight,
            mod_weight,
            add_sigmoid=add_sigmoid,
            assim_lr=assim_lr,
            assim_optimizer=assim_optimizer,
            assim_scheduler=assim_scheduler,
            assim_min_lr=assim_min_lr,
            grad_clip_norm=grad_clip_norm,
            amsgrad=amsgrad,
        )
        self.loss = LossVAE3DVarEXP(
            depth, factor,
            lat_weight,
            obs_weight,
            mod_weight
        )

    @staticmethod
    def init_params(config, full_config=None):
        path = Path(config['path'])
        device = full_config['device']
        creater = create_model[full_config['vae_type']]
        model = creater(config)
        model.from_pretrained(path)
        if 'add_sigmoid' not in config:
            config['add_sigmoid'] = False
        return {
            "model": VAEModel3Dvar._move_model_to_device(model, device),
            "depth": full_config["depth"],
            "factor": full_config["factor"],
            'lat_weight': full_config['lat_weight'],
            'obs_weight': full_config['obs_weight'],
            'mod_weight': full_config['mod_weight'],
            "add_sigmoid": config["add_sigmoid"],
            'assim_lr': full_config.get('assim_lr', 0.001),
            'assim_optimizer': full_config.get('assim_optimizer', 'adam'),
            'assim_scheduler': full_config.get('assim_scheduler'),
            'assim_min_lr': full_config.get('assim_min_lr', 0.0),
            'grad_clip_norm': full_config.get('grad_clip_norm'),
            'amsgrad': full_config.get('amsgrad', False),
        }


    @staticmethod
    def preprocess_tensors(tensors, config):
        x, x_b, y_b = tensors
        x = x.type(torch.float32)
        x_b = x_b.type(torch.float32)
        y_b = {
            key: y_b[key].type(torch.float32)
            for key in y_b
        }
        return [x, x_b, y_b]


    def fit(self, tensors, steps, meta):
        t = meta['day_of_year']
        def clone(y):
            return {
                key: y[key].clone().detach()
                for key in y
            }

        self.losses = []
        early_stopping = _init_early_stopping(meta.get("config", {}), steps)
        self.stopped_step = steps if early_stopping is not None else None
        x_init, x_b_init, y_init = tensors
        
        # init state
        y_obs = clone(y_init)
        self.x_b_init = x_b_init.clone().detach()
        self.y_init = clone(y_init)
        
        masks_y = {}
        # save masks
        for key in self.y_init:
            mask_x, masks_y[key] = self.get_valid_mask(
                self.x_b_init[:, 0], self.y_init[key]
            )
        
        # init lat
        lat_init, _ = self.model.encode(x_b_init)
        self.lat_init = lat_init.clone().detach()
        self.lat = lat_init.clone().detach()
        if hasattr(self.model, 'prepare_for_assimilation'):
            self.model.prepare_for_assimilation()
        
        self.lat.requires_grad_(True)
        optimizer = _make_latent_optimizer(
            [self.lat], self.assim_optimizer, self.lr, self.amsgrad
        )
        scheduler = _make_latent_scheduler(
            optimizer, self.assim_scheduler, steps, self.assim_min_lr
        )

        for step in range(1, steps + 1):
            optimizer.zero_grad()
            x_pred = self.model.decode(self.lat, t)
            if self.add_sigmoid:
                x_pred[:, 0] = torch.sigmoid(x_pred[:, 0])
            loss = 0
            batch_size = x_pred.shape[0]
            for i in range(batch_size):
                y_pred_i = {
                    key: x_pred[i, 0][masks_y[key][i]]
                    for key in masks_y
                }
                y_obs_i = {
                    key: y_obs[key][i][masks_y[key][i]].detach()
                    for key in y_obs
                } 
                loss += self.loss(
                    self.lat[i], lat_init[i].detach(), 
                    x_pred[i, 0][mask_x[i]], x_init[i, 0][mask_x[i]].detach(),
                    y_pred_i, y_obs_i
                )
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([self.lat], self.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = loss.detach().cpu().numpy().item()
            self.losses.append(loss_value)
            if _should_stop_early(early_stopping, loss_value, step):
                self.stopped_step = step
                print(f"Early stopping at step {step}/{steps}")
                break

        if early_stopping is not None:
            self.stopped_step = min(self.stopped_step, steps)

        self.x = self.model.decode(self.lat, t)
        if self.add_sigmoid:
            self.x[:, 0] = torch.sigmoid(self.x[:, 0])

    def get_obs(self):
        return self.y_init[0].detach().cpu()[None, ...]
