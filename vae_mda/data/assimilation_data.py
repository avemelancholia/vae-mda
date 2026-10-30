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
)
from .transform.var_transform import SIVarTransformer
from .transform.vae_transform import VaeTransformer

transformers = {
    SIVarTransformer.name: SIVarTransformer,
    VaeTransformer.name: VaeTransformer,
}


class Fields3DVARDS(Dataset):
    name = "Fields3DVARDS"
    
    def __init__(self, config, time=24, add_gauss=False):
        self.diagnostic_dir = None
        if config.get("diagnostic_dir"):
            self.diagnostic_dir = Path(config["diagnostic_dir"]) / "dataset"
            self.diagnostic_dir.mkdir(exist_ok=True, parents=True)
        self.data_init(config, time)
        config['add_gauss'] = add_gauss
        self.transformer = transformers[config['transform_name']](config)
        if self.diagnostic_dir is not None:
            self._save_dataset_metadata(config, time)
        
        
    def data_init(self, config, time):
        """
        self.ass_data: "sral" or "model"
        self.obs_shift: inx_obs - inx_back
        self.back_data
        self.obs_data
        """
        self.ass_data = config["ass_data"] # "sral" or "model"
        self.use_obs_confidence = bool(config.get("use_obs_confidence", False))
        if self.use_obs_confidence and self.ass_data != "sral":
            raise ValueError("use_obs_confidence is only supported for sral assimilation")
        self.model_target_alpha = float(config.get("model_target_alpha", 1.0))
        if not np.isfinite(self.model_target_alpha):
            raise ValueError("model_target_alpha must be finite")
        if self.model_target_alpha < 0.0 or self.model_target_alpha > 1.0:
            raise ValueError("model_target_alpha must be in [0, 1]")
        dataset_dir = Path(config["dataset_dir"])
        sat_dir = Path(config["sat_dir"])
        
        preds_path = dataset_dir / "preds"
        if "amsr_dir" in config:
            amsr_path = Path(config["amsr_dir"])
        else:
            amsr_path = sat_dir / "sats" / "amsr2"
        
        if "sral_dir" not in config:
            sral_path = sat_dir / "sats" / "sral"
        else:
            sral_path =  Path(config["sral_dir"])
        data = get_df(preds_path, sral_path, amsr_path)
        data = data[
            data.pred.str.startswith(
            str(preds_path) + nemo_folder[time])
        ]
        back_start_day, back_end_day, \
            obs_start_day, obs_end_day = self.date_prep(config)
        print(f"{back_start_day=}, {back_end_day=}, {obs_start_day=}, {obs_end_day=}")

        self.back_data = data[
            (data.date >= back_start_day) & (data.date <= back_end_day)
        ]
        self.obs_data = data[
            (data.date >= obs_start_day) & (data.date <= obs_end_day)
        ]
        if self.back_data.empty:
            raise ValueError(
                f"No background predictions in {preds_path} for "
                f"{back_start_day}..{back_end_day}"
            )
        required_observations = len(self.back_data) + self.obs_shift + 1
        if len(self.obs_data) < required_observations:
            raise ValueError(
                f"Observation/validation coverage is too short in {preds_path}: "
                f"need at least {required_observations} indexed days for "
                f"{len(self.back_data)} backgrounds, found {len(self.obs_data)}"
            )

    def date_prep(self, config):
        back_start_day = np.datetime64(config["back_start_day"])
        back_end_day = np.datetime64(config["back_end_day"])
        if self.ass_data == "sral":
            obs_start_day = back_start_day
            obs_end_day = back_end_day
        elif self.ass_data == "model":
            obs_start_day = np.datetime64(config["obs_start_day"])
            obs_end_day = np.datetime64(config["obs_end_day"])
            assert back_end_day-back_start_day == obs_end_day-obs_start_day, \
                "check dates"
        
        obs_end_day = obs_end_day + np.timedelta64(1, "D") # для валидации по следующему дню
        if "assimilation_range" in config:
            self.obs_shift = config["assimilation_range"] - 1 # для ассимиляции нескольких дней
        else:
            self.obs_shift = 0
        obs_start_day -= np.timedelta64(self.obs_shift, "D")
        return back_start_day, back_end_day, obs_start_day, obs_end_day

    def __len__(self):
        return len(self.back_data)

    @staticmethod
    def _paths_or_empty(paths):
        if paths is None:
            return []
        if hasattr(paths, "tolist"):
            paths = paths.tolist()
        if isinstance(paths, str):
            return [paths]
        return [str(path) for path in paths]

    @staticmethod
    def _tensor_to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _safe_date(date):
        return str(pd.to_datetime(date).date())

    @staticmethod
    def _finite_count(x):
        return int(np.sum(~np.isnan(x)))

    def _save_dataset_metadata(self, config, time):
        metadata = {
            "dataset_name": config.get("dataset_name"),
            "transform_name": config.get("transform_name"),
            "ass_data": config.get("ass_data"),
            "time": time,
            "assimilation_range": config.get("assimilation_range", 1),
            "obs_shift": self.obs_shift,
            "model_target_alpha": config.get("model_target_alpha", 1.0),
            "back_field": config.get("back_field"),
            "val_source": config.get("val_source"),
            "val_field": config.get("val_field"),
            "sral_indx": config.get("sral_indx"),
            "sral_obs_indx": config.get("sral_obs_indx"),
            "sral_val_sat_indx": config.get("sral_val_sat_indx"),
            "use_obs_confidence": config.get("use_obs_confidence", False),
            "sral_obs_confidence_indx": config.get("sral_obs_confidence_indx"),
            "date_ranges": {
                "back_start_day": config.get("back_start_day"),
                "back_end_day": config.get("back_end_day"),
                "obs_start_day": config.get("obs_start_day"),
                "obs_end_day": config.get("obs_end_day"),
            },
            "validation_map": [
                {
                    "name": "val_1",
                    "source": config.get("val_source", [None, None])[0],
                    "fields": config.get("val_field", [[], []])[0],
                    "day_relative_to_current_observation": 0,
                    "description": "same-day dense validation target",
                },
                {
                    "name": "val_2",
                    "source": config.get("val_source", [None, None])[1],
                    "fields": config.get("val_field", [[], []])[1],
                    "day_relative_to_current_observation": 1,
                    "description": "next-day sparse validation target",
                },
            ],
        }
        with open(self.diagnostic_dir / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _scale_model_target(self, target_field, background_field):
        if self.ass_data != "model" or self.model_target_alpha == 1.0:
            return target_field
        return background_field + self.model_target_alpha * (target_field - background_field)

    def _save_sample_diagnostic(
        self,
        sample_date,
        idx,
        obs_inx,
        back_row,
        val_row,
        next_row,
        back_field,
        back_mask,
        obs_field,
        obs_confidence,
        obs_diagnostics,
        val_field_1,
        val_field_2,
        val_field_1_raw,
        val_field_2_raw,
    ):
        if self.diagnostic_dir is None:
            return

        sample_path = self.diagnostic_dir / f"{sample_date}.npz"
        arrays = {
            "back_transformed": self._tensor_to_numpy(back_field),
            "back_mask": self._tensor_to_numpy(back_mask),
            "obs_combined": self._tensor_to_numpy(obs_field),
            "obs_combined_mask": np.isnan(self._tensor_to_numpy(obs_field)),
            "val_1": self._tensor_to_numpy(val_field_1),
            "val_1_mask": np.isnan(self._tensor_to_numpy(val_field_1)),
            "val_2": self._tensor_to_numpy(val_field_2),
            "val_2_mask": np.isnan(self._tensor_to_numpy(val_field_2)),
            "val_1_raw": val_field_1_raw,
            "val_2_raw": val_field_2_raw,
        }
        if obs_confidence is not None:
            arrays["obs_confidence"] = self._tensor_to_numpy(obs_confidence)
        for obs_meta in obs_diagnostics:
            offset = obs_meta["offset"]
            arrays[f"obs_{offset}_raw"] = obs_meta["raw"]
            arrays[f"obs_{offset}_transformed"] = obs_meta["transformed"]
            arrays[f"obs_{offset}_mask"] = obs_meta["mask"]
            if "confidence" in obs_meta:
                arrays[f"obs_{offset}_confidence"] = obs_meta["confidence"]
        np.savez_compressed(sample_path, **arrays)

        obs_metadata = []
        for obs_meta in obs_diagnostics:
            obs_metadata.append({
                "offset": obs_meta["offset"],
                "relative_day": -obs_meta["offset"],
                "date": obs_meta["date"],
                "pred_path": obs_meta["pred_path"],
                "sral_paths": obs_meta["sral_paths"],
                "raw_finite_count": obs_meta["raw_finite_count"],
                "transformed_finite_count": obs_meta["transformed_finite_count"],
                "confidence_finite_count": obs_meta.get("confidence_finite_count"),
            })

        metadata = {
            "sample_date": sample_date,
            "idx": int(idx),
            "obs_inx": int(obs_inx),
            "ass_data": self.ass_data,
            "model_target_alpha": self.model_target_alpha,
            "back": {
                "date": self._safe_date(back_row.date),
                "pred_path": str(back_row.pred),
                "finite_count": self._finite_count(arrays["back_transformed"]),
            },
            "observations_used_for_assimilation": obs_metadata,
            "val_1": {
                "date": self._safe_date(val_row.date),
                "source": "amsr2" if self.ass_data == "sral" else "model",
                "amsr_paths": self._paths_or_empty(val_row.amsr),
                "pred_path": str(val_row.pred),
                "finite_count": self._finite_count(arrays["val_1"]),
                "meaning": "same-day validation target",
            },
            "val_2": {
                "date": self._safe_date(next_row.date),
                "source": "sral" if self.ass_data == "sral" else "track_imitation",
                "sral_paths": self._paths_or_empty(next_row.sral),
                "pred_path": str(next_row.pred),
                "finite_count": self._finite_count(arrays["val_2"]),
                "meaning": "next-day sparse validation target",
            },
            "npz_file": str(sample_path),
        }
        with open(self.diagnostic_dir / f"{sample_date}.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_obs(self, obs_inx, shape, collect_diagnostics=False, model_background_field=None):
        obs_fields = torch.full(shape, np.nan)
        obs_confidences = torch.full(shape, np.nan) if self.use_obs_confidence else None
        diagnostics = []

        for i in range(self.obs_shift+1):
            row = self.obs_data.iloc[obs_inx-i, :]
            if i==0:
                print(f"row_obs.data = {row.date}")
            print(f"{row.sral=}")
            obs_dirs = self._paths_or_empty(row.sral)
            if not obs_dirs:
                print("empty obs_dirs")
            obs_field = np.zeros(shape)
            obs_field.fill(np.nan)
            for dir_ in obs_dirs:
                sral_field = np.load(dir_)
                if sral_field.shape!=shape:
                    obs_field = np.zeros(sral_field.shape)
                    obs_field.fill(np.nan)
                obs_field[np.isnan(obs_field)] = sral_field[np.isnan(obs_field)]
            obs_field_raw = obs_field.copy()
            obs_confidence = None
            if self.use_obs_confidence:
                obs_confidence = self.transformer.obs_confidence_transform(obs_field_raw)
            obs_field = self.transformer.obs_transform(obs_field)
            if self.ass_data == "model":
                sral_mask = np.isnan(obs_field.cpu().numpy())
                model_field = np.load(row.pred)[:, 23]
                if self.model_target_alpha != 1.0:
                    if model_background_field is None:
                        raise ValueError(
                            "model_background_field is required when model_target_alpha != 1"
                        )
                    model_field = self._scale_model_target(model_field, model_background_field)
                obs_field = self.transformer.obs_model_transform(
                    model_field, sral_mask,
                )
            if obs_fields.shape!=obs_field.shape:
                obs_fields = torch.full(obs_field.shape, np.nan)
                if self.use_obs_confidence:
                    obs_confidences = torch.full(obs_field.shape, np.nan)
            fill_mask = torch.isnan(obs_fields) & ~torch.isnan(obs_field)
            obs_fields[fill_mask] = obs_field[fill_mask]
            if self.use_obs_confidence:
                obs_confidences[fill_mask] = obs_confidence[fill_mask]
            if collect_diagnostics:
                obs_field_np = self._tensor_to_numpy(obs_field)
                diagnostic = {
                    "offset": i,
                    "date": self._safe_date(row.date),
                    "pred_path": str(row.pred),
                    "sral_paths": obs_dirs,
                    "raw": obs_field_raw,
                    "transformed": obs_field_np,
                    "mask": np.isnan(obs_field_np),
                    "raw_finite_count": self._finite_count(obs_field_raw),
                    "transformed_finite_count": self._finite_count(obs_field_np),
                }
                if obs_confidence is not None:
                    obs_confidence_np = self._tensor_to_numpy(obs_confidence)
                    diagnostic["confidence"] = obs_confidence_np
                    diagnostic["confidence_finite_count"] = self._finite_count(obs_confidence_np)
                diagnostics.append(diagnostic)
        if collect_diagnostics:
            if self.use_obs_confidence:
                return obs_fields, obs_confidences, diagnostics
            return obs_fields, diagnostics
        if self.use_obs_confidence:
            return obs_fields, obs_confidences
        return obs_fields


    def __getitem__(self, idx):
        """
            Return:
            (
                (back_field, back_field, obs_field) shape: [ch, h, w], [ch, h, w], [h, w]
                val_field, shape: [h, w]
                {"date": str(day.date())} str
            )
        """
        print("standart load")
        back_row = self.back_data.iloc[idx, :]

        day = pdl.datetime(
            back_row.date.year,
            back_row.date.month,
            back_row.date.day,
        )
        print(f"{back_row.date=}")
        # backward
        back_field_raw = np.load(back_row.pred)[:, 23]
        back_field, back_mask = self.transformer.transform(back_field_raw)

        obs_inx = idx+self.obs_shift
        val_row = self.obs_data.iloc[obs_inx, :]
        
        if self.ass_data == "sral":
            print(val_row.amsr[0])
            val_field = np.load(val_row.amsr[0])
            val_field_1_raw = val_field.copy()
            val_field_1 = self.transformer.val_transform(val_field)
        elif self.ass_data == "model":
            val_field = np.load(val_row.pred)[:, 23]
            val_field = self._scale_model_target(val_field, back_field_raw)
            val_field_1_raw = val_field.copy()
            val_field_1 = self.transformer.val_model_transform(val_field)
        if self.diagnostic_dir is not None:
            if self.use_obs_confidence:
                obs_field, obs_confidence, obs_diagnostics = self.load_obs(
                    obs_inx, val_field_1[0].shape, collect_diagnostics=True,
                    model_background_field=back_field_raw,
                )
            else:
                obs_field, obs_diagnostics = self.load_obs(
                    obs_inx, val_field_1[0].shape, collect_diagnostics=True,
                    model_background_field=back_field_raw,
                )
                obs_confidence = None
        else:
            if self.use_obs_confidence:
                obs_field, obs_confidence = self.load_obs(
                    obs_inx, val_field_1[0].shape,
                    model_background_field=back_field_raw,
                )
            else:
                obs_field = self.load_obs(
                    obs_inx, val_field_1[0].shape,
                    model_background_field=back_field_raw,
                )
                obs_confidence = None
            obs_diagnostics = []

        # next SRAL
        row = self.obs_data.iloc[obs_inx+1, :]
        dirs = self._paths_or_empty(row.sral)
    
        val_field_2 = np.zeros(val_field_1[0].shape)
        val_field_2.fill(np.nan)
        for dir_ in dirs:
            field = np.load(dir_)
            if field.shape!=val_field_2.shape:
                val_field_2 = np.zeros(field.shape)
                val_field_2.fill(np.nan)
            val_field_2[np.isnan(val_field_2)] = field[np.isnan(val_field_2)]
        val_field_2_raw = val_field_2.copy()
        val_field_2 = self.transformer.val_sat_transform(val_field_2)

        if self.ass_data == "model":
            val_field_2_mask = torch.isnan(val_field_2)
            val_field_2 = val_field_1[0].clone()[None, ...]
            val_field_2[val_field_2_mask] = float('nan')

        self._save_sample_diagnostic(
            str(day.date()), idx, obs_inx, back_row, val_row, row,
            back_field, back_mask, obs_field, obs_confidence, obs_diagnostics,
            val_field_1, val_field_2, val_field_1_raw, val_field_2_raw,
        )
        tensors = (back_field, back_field, obs_field)
        if self.use_obs_confidence:
            tensors = (back_field, back_field, obs_field, obs_confidence)
        return (
            tensors,
            val_field_1, val_field_2,
            {
                "date": str(day.date()), "back_mask": back_mask,
                "day_of_year": day.day_of_year,
            },
        )


class Fields3DVARSampledDS(Fields3DVARDS):
    name = "Fields3DVARSampledDS"

    def __init__(self, config, time=24):
        super().__init__(config, time=time)
        self.sample_indices = self._build_sample_indices(config)

    @staticmethod
    def _date_key(date):
        return pd.to_datetime(date).date().isoformat()

    def _build_sample_indices(self, config):
        if "sample_dates" not in config:
            raise ValueError("Fields3DVARSampledDS requires sample_dates in config")

        date_to_idx = {
            self._date_key(row.date): idx
            for idx, row in enumerate(self.back_data.itertuples())
        }
        sample_indices = []
        missing_dates = []
        for date in config["sample_dates"]:
            key = self._date_key(date)
            if key not in date_to_idx:
                missing_dates.append(key)
                continue
            sample_indices.append(date_to_idx[key])

        if missing_dates:
            available_start = self._date_key(self.back_data.iloc[0].date)
            available_end = self._date_key(self.back_data.iloc[-1].date)
            raise ValueError(
                f"sample_dates not found: {missing_dates}. "
                f"Available back_data range: {available_start}..{available_end}"
            )
        return sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.sample_indices[idx])


class Fields3DVAREXPDS(Fields3DVARDS):
    name = "Fields3DVAREXPDS"
    def __init__(self, config, time=24):
        self.depth = config["depth"]
        super().__init__(config, time=24)

    def date_prep(self, config):
        back_start_day = np.datetime64(config["back_start_day"])
        back_end_day = np.datetime64(config["back_end_day"])
        if self.ass_data == "sral":
            obs_start_day = back_start_day
            obs_end_day = back_end_day
        elif self.ass_data == "model":
            obs_start_day = np.datetime64(config["obs_start_day"])
            obs_end_day = np.datetime64(config["obs_end_day"])
            assert back_end_day-back_start_day == obs_end_day-obs_start_day, \
                "check dates"
        obs_end_day = obs_end_day + np.timedelta64(1, "D") # для валидации по следующему дню

        if "depth" in config:
            self.depth = config["depth"]
            self.obs_shift = self.depth
        else:
            self.depth = 1
            self.obs_shift = 0
    
        obs_start_day -= np.timedelta64(self.depth, "D")
        return back_start_day, back_end_day, obs_start_day, obs_end_day

    def __len__(self):
        return len(self.back_data)

    def load_obs(self, obs_inx, shape, collect_diagnostics=False, model_background_field=None):
        obs_fields = {}
        for i in range(0, self.depth):
            row = self.obs_data.iloc[obs_inx-i, :]
            if i==0:
                print(f"row_obs.data = {row.date}")
            if row.sral is not None:
                obs_dirs = row.sral.tolist()
            else: 
                obs_dirs = []
                print("empty obs_dirs")
            
            obs_field = np.zeros(shape)
            obs_field.fill(np.nan)
            
            for dir_ in obs_dirs:
                sral_field = np.load(dir_)
                if sral_field.shape!=shape:
                    obs_field = np.zeros(sral_field.shape)
                    obs_field.fill(np.nan)
                obs_field[np.isnan(obs_field)] = sral_field[np.isnan(obs_field)]
            obs_field = self.transformer.obs_transform(obs_field)
            if self.ass_data == "model":
                sral_mask = np.isnan(obs_field.cpu().numpy())
                model_field = np.load(row.pred)[:, 23]
                if self.model_target_alpha != 1.0:
                    if model_background_field is None:
                        raise ValueError(
                            "model_background_field is required when model_target_alpha != 1"
                        )
                    model_field = self._scale_model_target(model_field, model_background_field)
                obs_field = self.transformer.obs_model_transform(
                    model_field, sral_mask,
                )
            obs_fields[i] = obs_field
        if collect_diagnostics:
            return obs_fields, []
        return obs_fields

    def __getitem__(self, idx):
        """
            Return:
            (
                (back_field, back_field, obs_field/100) shape: [ch, h, w], [ch, h, w], dict {t_delta:[h, w]}
                val_field, shape: [h, w]
                {"date": str(day.date())} 
            )
        """
        print("exp load")
        return super().__getitem__(idx)
