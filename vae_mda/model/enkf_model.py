import numpy as np
import torch

from .base_model import AssimilationModel
from .var.polynomial_matrix import haversine_distance


class ModelEnKF(AssimilationModel):
    name = "enkf_baseline"

    def __init__(
        self,
        ensemble_size,
        background_std,
        obs_std,
        inflation,
        clip_min,
        clip_max,
        seed,
        localization=None,
        lat_path=None,
        lon_path=None,
        mask_path=None,
        limit=None,
        perturbations="iid",
        perturbation_radius=3,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.background_std = background_std
        self.obs_std = obs_std
        self.inflation = inflation
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.seed = seed
        self.localization = localization
        self.limit = limit
        self.perturbations = perturbations
        self.perturbation_radius = perturbation_radius
        self.losses = []

        self.lat = None
        self.lon = None
        self.mask = None
        if self.localization == "polynomial":
            if None in (lat_path, lon_path, mask_path, limit):
                raise ValueError(
                    "polynomial localization requires lat_path, lon_path, mask_path, and limit"
                )
            self.lat = np.load(lat_path)
            self.lon = np.load(lon_path)
            self.mask = np.load(mask_path)

    @staticmethod
    def init_params(config, full_config=None):
        return {
            "ensemble_size": config["ensemble_size"],
            "background_std": config["background_std"],
            "obs_std": config["obs_std"],
            "inflation": config.get("inflation", 1.0),
            "clip_min": config.get("clip_min", 0.0),
            "clip_max": config.get("clip_max", 1.0),
            "seed": config.get("seed", 42),
            "localization": config.get("localization"),
            "lat_path": config.get("lat_path"),
            "lon_path": config.get("lon_path"),
            "mask_path": config.get("mask_path"),
            "limit": config.get("limit"),
            "perturbations": config.get("perturbations", "iid"),
            "perturbation_radius": config.get("perturbation_radius", 3),
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

    def _inflate(self, ensemble):
        if self.inflation == 1.0:
            return ensemble
        mean = ensemble.mean(axis=0, keepdims=True)
        return mean + self.inflation * (ensemble - mean)

    def _polynomial_weights(self, state_lats, state_lons, obs_idx):
        distances = haversine_distance(
            state_lats,
            state_lons,
            state_lats[obs_idx:obs_idx + 1],
            state_lons[obs_idx:obs_idx + 1],
        )[:, 0]
        weights = np.zeros_like(distances, dtype=np.float32)

        mask = distances <= self.limit
        ratio = distances[mask] / self.limit
        weights[mask] = (
            -(ratio ** 5) / 4
            + (ratio ** 4) / 2
            + 5 * (ratio ** 3) / 8
            - 5 * (ratio ** 2) / 3
            + 1
        )

        mask = (distances > self.limit) & (distances <= 2 * self.limit)
        ratio = distances[mask] / self.limit
        weights[mask] = (
            (ratio ** 5) / 12
            - (ratio ** 4) / 2
            + 5 * (ratio ** 3) / 8
            + 5 * (ratio ** 2) / 3
            - 5 * ratio
            + 4
            - 2 / (3 * (ratio + 1e-9))
        )
        return weights

    def _build_perturbation_kernel(self):
        radius = int(self.perturbation_radius)
        if radius <= 0:
            return np.ones((1, 1), dtype=np.float32)

        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
        distances = np.sqrt(dx ** 2 + dy ** 2)
        weights = np.zeros_like(distances, dtype=np.float32)

        mask = distances <= radius
        ratio = distances[mask] / radius
        weights[mask] = (
            -(ratio ** 5) / 4
            + (ratio ** 4) / 2
            + 5 * (ratio ** 3) / 8
            - 5 * (ratio ** 2) / 3
            + 1
        )
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("perturbation kernel has zero weight")
        return weights / weight_sum

    @staticmethod
    def _shift_field(field, di, dj):
        shifted = np.zeros_like(field)

        if di >= 0:
            src_i = slice(0, field.shape[0] - di)
            dst_i = slice(di, field.shape[0])
        else:
            src_i = slice(-di, field.shape[0])
            dst_i = slice(0, field.shape[0] + di)

        if dj >= 0:
            src_j = slice(0, field.shape[1] - dj)
            dst_j = slice(dj, field.shape[1])
        else:
            src_j = slice(-dj, field.shape[1])
            dst_j = slice(0, field.shape[1] + dj)

        shifted[dst_i, dst_j] = field[src_i, src_j]
        return shifted

    def _smooth_masked_field(self, field, valid_mask, kernel):
        radius = kernel.shape[0] // 2
        smoothed = np.zeros_like(field, dtype=np.float32)
        norm = np.zeros_like(field, dtype=np.float32)

        for i, di in enumerate(range(-radius, radius + 1)):
            for j, dj in enumerate(range(-radius, radius + 1)):
                weight = kernel[i, j]
                if weight == 0:
                    continue
                shifted_values = self._shift_field(field, di, dj)
                shifted_mask = self._shift_field(valid_mask.astype(np.float32), di, dj)
                smoothed += weight * shifted_values * shifted_mask
                norm += weight * shifted_mask

        output = np.zeros_like(field, dtype=np.float32)
        good = norm > 0
        output[good] = smoothed[good] / norm[good]
        output[~valid_mask] = 0.0
        return output

    def _sample_background_ensemble(self, x_b, x_mask, rng):
        x_b_state = x_b[x_mask].astype(np.float32)
        state_size = x_b_state.shape[0]

        if self.perturbations == "iid":
            ensemble = x_b_state[None, :] + self.background_std * rng.standard_normal(
                size=(self.ensemble_size, state_size)
            ).astype(np.float32)
            return self._inflate(ensemble)

        if self.perturbations != "correlated":
            raise ValueError(f"Unknown perturbations mode: {self.perturbations}")

        kernel = self._build_perturbation_kernel()
        ensemble = np.zeros((self.ensemble_size, state_size), dtype=np.float32)
        valid_mask = x_mask.astype(bool)

        for member_idx in range(self.ensemble_size):
            noise = rng.standard_normal(size=x_b.shape).astype(np.float32)
            noise[~valid_mask] = 0.0
            correlated_noise = self._smooth_masked_field(noise, valid_mask, kernel)

            state_noise = correlated_noise[x_mask]
            state_std = float(state_noise.std())
            if state_std > 1e-8:
                state_noise = state_noise / state_std

            ensemble[member_idx] = x_b_state + self.background_std * state_noise

        return self._inflate(ensemble)

    def fit(self, tensors, steps=None, meta=None):
        x_init, x_b_init, y_init = tensors
        if x_init.shape[0] != 1:
            raise ValueError("enkf_baseline currently supports batch_size=1 only")
        if x_init.shape[1] != 1:
            raise ValueError("enkf_baseline currently supports siconc-only inputs")

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
            self.losses = [0.0]
            return

        rng = np.random.default_rng(self.seed)
        obs_values = y[y_mask].astype(np.float32)
        state_lats = None
        state_lons = None

        if self.localization == "polynomial":
            if self.lat.shape != x_b.shape or self.lon.shape != x_b.shape or self.mask.shape != x_b.shape:
                raise ValueError("localization metadata shape must match the siconc field shape")
            state_lats = self.lat[x_mask].astype(np.float32)
            state_lons = self.lon[x_mask].astype(np.float32)

        state_indices = np.full(x_b.shape, -1, dtype=np.int64)
        state_indices[x_mask] = np.arange(state_size)
        obs_indices = state_indices[y_mask]

        ensemble = self._sample_background_ensemble(x_b, x_mask, rng)

        innovations_before = []
        innovations_after = []
        denom_floor = 1e-8

        for obs_idx, obs_value in zip(obs_indices, obs_values):
            forecast_obs = ensemble[:, obs_idx]
            innovations_before.append(float((forecast_obs.mean() - obs_value) ** 2))

            ensemble_mean = ensemble.mean(axis=0)
            anomalies = ensemble - ensemble_mean
            obs_anomalies = forecast_obs - forecast_obs.mean()

            if self.ensemble_size > 1:
                scale = 1.0 / (self.ensemble_size - 1)
            else:
                scale = 1.0

            cov_xh = anomalies.T @ obs_anomalies * scale
            var_h = float(obs_anomalies @ obs_anomalies * scale)
            gain = cov_xh / max(var_h + self.obs_std ** 2, denom_floor)
            if self.localization == "polynomial":
                gain = gain * self._polynomial_weights(state_lats, state_lons, obs_idx)

            perturbed_obs = obs_value + self.obs_std * rng.standard_normal(
                self.ensemble_size
            ).astype(np.float32)
            ensemble = ensemble + (perturbed_obs - forecast_obs)[:, None] * gain[None, :]

            updated_obs = ensemble[:, obs_idx]
            innovations_after.append(float((updated_obs.mean() - obs_value) ** 2))

        analysis_state = ensemble.mean(axis=0)
        analysis[x_mask] = np.clip(analysis_state, self.clip_min, self.clip_max)
        self.x = torch.tensor(
            analysis,
            dtype=self.x_b_init.dtype,
            device=self.x_b_init.device,
        )

        forecast_rmse = float(np.sqrt(np.mean(innovations_before)))
        analysis_rmse = float(np.sqrt(np.mean(innovations_after)))
        self.losses = [forecast_rmse, analysis_rmse]

    def get_x(self):
        return self.x_b_init[None, None, ...], self.x[None, None, ...]

    def get_obs(self):
        return self.y_init[None, ...]
