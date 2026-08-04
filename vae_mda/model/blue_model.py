import numpy as np
import torch

from .base_model import AssimilationModel


class ModelBLUE(AssimilationModel):
    name = "blue_baseline"

    def __init__(
        self,
        covariance_matrix,
        obs_std,
        regularization,
        clip_min,
        clip_max,
    ):
        super().__init__()
        self.covariance_matrix = covariance_matrix
        self.obs_std = obs_std
        self.regularization = regularization
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.losses = []

    @staticmethod
    def init_params(config, full_config=None):
        inv_matrix = np.load(config["inv_corr_matrix_path"])
        covariance_matrix = np.linalg.inv(inv_matrix).astype(np.float32)
        covariance_matrix *= config["inv_corr_matrix_sigma2"]
        return {
            "covariance_matrix": covariance_matrix,
            "obs_std": config["obs_std"],
            "regularization": config.get("regularization", 1e-6),
            "clip_min": config.get("clip_min", 0.0),
            "clip_max": config.get("clip_max", 1.0),
        }

    @staticmethod
    def preprocess_tensors(tensors, config):
        x, x_b, y_b = tensors
        return [x, x_b, y_b]

    @staticmethod
    def get_valid_mask(x, y):
        x_mask = ~x.isnan()
        y_mask = x_mask & (~y.isnan())
        return x_mask, y_mask

    def fit(self, tensors, steps=None, meta=None):
        x_init, x_b_init, y_init = tensors
        if x_init.shape[0] != 1:
            raise ValueError("blue_baseline currently supports batch_size=1 only")
        if x_init.shape[1] != 1:
            raise ValueError("blue_baseline currently supports siconc-only inputs")

        self.x_b_init = x_b_init[0, 0].clone().detach()
        self.y_init = y_init[0].clone().detach()

        x_b = self.x_b_init.detach().cpu().numpy()
        y = self.y_init.detach().cpu().numpy()
        x_mask, y_mask = self.get_valid_mask(self.x_b_init, self.y_init)
        x_mask = x_mask.detach().cpu().numpy()
        y_mask = y_mask.detach().cpu().numpy()

        state_size = int(x_mask.sum())
        obs_size = int(y_mask.sum())
        analysis = np.full_like(x_b, np.nan, dtype=np.float32)
        analysis[x_mask] = x_b[x_mask]

        if obs_size == 0 or state_size == 0:
            self.x = torch.tensor(
                analysis,
                dtype=self.x_b_init.dtype,
                device=self.x_b_init.device,
            )
            self.losses = [0.0, 0.0]
            return

        if self.covariance_matrix.shape != (state_size, state_size):
            raise ValueError(
                f"Covariance matrix shape {self.covariance_matrix.shape} does not match state size {(state_size, state_size)}"
            )

        state_indices = np.full(x_b.shape, -1, dtype=np.int64)
        state_indices[x_mask] = np.arange(state_size)
        obs_idx = state_indices[y_mask]

        x_b_state = x_b[x_mask].astype(np.float32)
        y_obs = y[y_mask].astype(np.float32)
        innovation = y_obs - x_b_state[obs_idx]

        b_xo = self.covariance_matrix[:, obs_idx]
        b_oo = self.covariance_matrix[np.ix_(obs_idx, obs_idx)]
        solve_matrix = b_oo + (
            self.obs_std ** 2 + self.regularization
        ) * np.eye(obs_size, dtype=np.float32)
        weights = np.linalg.solve(solve_matrix, innovation)
        analysis_state = x_b_state + b_xo @ weights

        analysis[x_mask] = np.clip(analysis_state, self.clip_min, self.clip_max)
        self.x = torch.tensor(
            analysis,
            dtype=self.x_b_init.dtype,
            device=self.x_b_init.device,
        )

        forecast_rmse = float(np.sqrt(np.mean(innovation ** 2)))
        analysis_innovation = analysis[x_mask][obs_idx] - y_obs
        analysis_rmse = float(np.sqrt(np.mean(analysis_innovation ** 2)))
        self.losses = [forecast_rmse, analysis_rmse]

    def get_x(self):
        return self.x_b_init[None, None, ...], self.x[None, None, ...]

    def get_obs(self):
        return self.y_init[None, ...]
