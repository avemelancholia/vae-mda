import torch
import csv
import json
import numpy as np
import time
from pathlib import Path

from .logger.assimilation_logger import AssimilationCML
from .utils.restart import (
    assimilate_restarts_physical,
    assimilate_restarts_physical_ocean,
    assimilate_restarts_physical_ocean_bounded,
    assimilate_restarts_naively,
)


class AssimilationHandler:
    def __init__(
        self, config, dataloader, model_class, model_config, data_config
    ):
        self.config = config
        self.dataloader = dataloader
        self.model_class = model_class
        self.model_config = model_config
        self.data_config = data_config
        self.logger = AssimilationCML(config, model_config, data_config)
        
        self.dataset = self._base_dataset(self.dataloader.dataset)
        self.transformer = self.dataset.transformer
        self.diagnostic_dir = None
        if config.get("diagnostic_dir"):
            self.diagnostic_dir = Path(config["diagnostic_dir"])
            self.analysis_dir = self.diagnostic_dir / "analysis"
            self.analysis_dir.mkdir(exist_ok=True, parents=True)
            self.summary_path = self.diagnostic_dir / "summary.csv"
            self._write_run_metadata(config, model_config, data_config)
            self._init_summary()
        torch.set_default_device(config["device"])
        torch.set_default_dtype(torch.float32)

    @staticmethod
    def _base_dataset(dataset):
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    @staticmethod
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(key): AssimilationHandler._json_safe(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AssimilationHandler._json_safe(value) for value in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @staticmethod
    def _finite_count(x):
        return int(np.sum(~np.isnan(x)))

    @staticmethod
    def _nan_mae(pred, gt):
        mask = (~np.isnan(pred)) & (~np.isnan(gt))
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs(pred[mask] - gt[mask])))

    @staticmethod
    def _nan_rmse(pred, gt):
        mask = (~np.isnan(pred)) & (~np.isnan(gt))
        if not np.any(mask):
            return np.nan
        return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))

    def _sync_device(self):
        if str(self.config.get("device", "")).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _write_run_metadata(self, config, model_config, data_config):
        metadata = {
            "config": self._json_safe(config),
            "model_config": self._json_safe(model_config),
            "data_config": self._json_safe(data_config),
            "diagnostic_notes": {
                "val_1": "same-day validation target from data_config.val_source[0]",
                "val_2": "next-day validation target from data_config.val_source[1]",
                "obs": "combined observations used for assimilation after transform/detransform",
                "analysis": "model output after detransform",
            },
        }
        self.diagnostic_dir.mkdir(exist_ok=True, parents=True)
        with open(self.diagnostic_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _init_summary(self):
        if self.summary_path.exists():
            return
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "date",
                "obs_count",
                "val_1_count",
                "val_2_count",
                "mean_abs_delta_siconc",
                "val_1_model_mae",
                "val_1_assim_mae",
                "val_1_model_rmse",
                "val_1_assim_rmse",
                "val_2_model_mae",
                "val_2_assim_mae",
                "val_2_model_rmse",
                "val_2_assim_rmse",
            ])

    def _save_analysis_diagnostic(self, date, init_x, x, obs, val_1, val_2, losses):
        if self.diagnostic_dir is None:
            return

        delta = x - init_x
        val_1_gt = val_1[0]
        val_2_gt = val_2[0]
        init_siconc = init_x[0]
        analysis_siconc = x[0]

        if isinstance(losses, dict):
            losses_array = np.asarray(
                [value for _, value in losses.get("assimilation_loss", [])],
                dtype=np.float32,
            )
        else:
            losses_array = np.asarray(losses, dtype=np.float32)

        np.savez_compressed(
            self.analysis_dir / f"{date}.npz",
            init_x=init_x,
            analysis=x,
            obs=obs,
            val_1=val_1,
            val_2=val_2,
            delta=delta,
            obs_mask=np.isnan(obs),
            val_1_mask=np.isnan(val_1),
            val_2_mask=np.isnan(val_2),
            analysis_minus_val_1_siconc=analysis_siconc - val_1_gt,
            analysis_minus_val_2_siconc=analysis_siconc - val_2_gt,
            init_minus_val_1_siconc=init_siconc - val_1_gt,
            init_minus_val_2_siconc=init_siconc - val_2_gt,
            losses=losses_array,
        )

        summary = {
            "date": date,
            "obs_count": self._finite_count(obs),
            "val_1_count": self._finite_count(val_1),
            "val_2_count": self._finite_count(val_2),
            "mean_abs_delta_siconc": float(np.nanmean(np.abs(delta[0]))),
            "val_1_model_mae": self._nan_mae(init_siconc, val_1_gt),
            "val_1_assim_mae": self._nan_mae(analysis_siconc, val_1_gt),
            "val_1_model_rmse": self._nan_rmse(init_siconc, val_1_gt),
            "val_1_assim_rmse": self._nan_rmse(analysis_siconc, val_1_gt),
            "val_2_model_mae": self._nan_mae(init_siconc, val_2_gt),
            "val_2_assim_mae": self._nan_mae(analysis_siconc, val_2_gt),
            "val_2_model_rmse": self._nan_rmse(init_siconc, val_2_gt),
            "val_2_assim_rmse": self._nan_rmse(analysis_siconc, val_2_gt),
        }
        with open(self.analysis_dir / f"{date}.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(self.summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([summary[key] for key in [
                "date",
                "obs_count",
                "val_1_count",
                "val_2_count",
                "mean_abs_delta_siconc",
                "val_1_model_mae",
                "val_1_assim_mae",
                "val_1_model_rmse",
                "val_1_assim_rmse",
                "val_2_model_mae",
                "val_2_assim_mae",
                "val_2_model_rmse",
                "val_2_assim_rmse",
            ]])

    @staticmethod
    def _finite_mae_rmse(pred, gt):
        mask = (~np.isnan(pred)) & (~np.isnan(gt))
        if not np.any(mask):
            return np.nan, np.nan
        diff = pred[mask] - gt[mask]
        return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))

    def _build_validation_callback(self, val_1):
        val_1_np = val_1.detach().cpu().numpy()
        back_fields = list(self.logger.back_field)
        val_fields = list(self.logger.val_field[0])

        def validation_callback(x_pred):
            x_pred_np = self.transformer.detransform(x_pred.detach().clone())
            metrics = {}
            for field in back_fields:
                if field not in val_fields:
                    continue
                pred_idx = back_fields.index(field)
                val_idx = val_fields.index(field)
                mae, rmse = self._finite_mae_rmse(
                    x_pred_np[:, pred_idx],
                    val_1_np[:, val_idx],
                )
                metrics[f"val_{field}_mae"] = mae
                metrics[f"val_{field}_rmse"] = rmse
            return metrics

        return validation_callback


    def run_assimilation(self):
        
        params = self.model_class.init_params(self.model_config, self.config)
        model = self.model_class(**params)

        for tensors, val_1, val_2, meta in self.dataloader:
            print(f'Date {meta["date"]} is in work.')
            day_start = time.perf_counter()
            timings = {}

            t0 = time.perf_counter()
            tensors = model.preprocess_tensors(
                tensors, self.config,
            )
            self._sync_device()
            timings["preprocess"] = time.perf_counter() - t0
            if "steps" not in self.config:
                self.config["steps"] = None

            fit_meta = dict(meta)
            fit_meta["config"] = self.config
            if self.config.get("track_validation_during_fit", False):
                fit_meta["validation_callback"] = self._build_validation_callback(val_1)
            self._sync_device()
            t0 = time.perf_counter()
            model.fit(tensors, self.config["steps"], fit_meta)
            self._sync_device()
            timings["fit"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            init_x, x = model.get_x()
            obs = model.get_obs()
            
            init_x = self.transformer.detransform(init_x) #, meta["back_mask"])
            x = self.transformer.detransform(x) #, meta["back_mask"])
            obs = self.transformer.obs_detransform(obs)
            self._sync_device()
            timings["detransform"] = time.perf_counter() - t0

            batch_size = x.shape[0]
            for i in range(batch_size):
                date = str(meta["date"][i])
                val_1_i = val_1[i].cpu().numpy()
                val_2_i = val_2[i].cpu().numpy()
                t0 = time.perf_counter()
                self._save_analysis_diagnostic(
                    date, init_x[i], x[i], obs[i], val_1_i, val_2_i,
                    model.losses,
                )
                timings["diagnostics"] = time.perf_counter() - t0
                self.logger.log(
                    model.losses, date,
                    init_x[i], x[i], obs[i],
                    val_1[i].cpu().numpy(), val_2[i].cpu().numpy(),
                    stopped_step=getattr(model, "stopped_step", None),
                    best_validation_step=getattr(model, "best_validation_step", None),
                    timings=timings,
                    day_start=day_start,
                )

        self.logger.close()


class InferenceHandler:
    def __init__(
        self, config, dataloader, model_class, model_config, data_config
    ):
        self.config = config
        self.dataloader = dataloader
        self.model_class = model_class
        self.model_config = model_config
        self.data_config = data_config
        
        self.transformer = self.dataloader.dataset.transformer
        torch.set_default_device(config["device"])
        torch.set_default_dtype(torch.float32)


    def run_assimilation(self):
        
        params = self.model_class.init_params(self.model_config, self.config)
        model = self.model_class(**params)

        for tensors, meta in self.dataloader:
            print(f'Date {meta["date"]} is in work.')

            tensors = model.preprocess_tensors(
                tensors, self.config,
            )
            if "steps" not in self.config:
                self.config["steps"] = None

            fit_meta = dict(meta)
            fit_meta["config"] = self.config
            model.fit(tensors, self.config["steps"], fit_meta)

            init_x, x = model.get_x()
            obs = model.get_obs()
            
            init_x = self.transformer.detransform(init_x) #, meta["back_mask"])
            x = self.transformer.detransform(x) #, meta["back_mask"])
            obs = self.transformer.obs_detransform(obs)

            batch_size = x.shape[0]

        restart_dir = Path(self.config['restart_dir'])/f"run_{self.config['date']}T00:00:00+00:00"/"nemo_restart"
        
        if self.config['assim_type'] == 'physical':
            assimilate_restarts_physical(restart_dir, x[0][0], x[0][1], self.config['domain'])
        elif self.config['assim_type'] == 'physical_ocean':
            back_fields = list(self.data_config.get("back_field", []))
            required_fields = ("siconc", "sithic", "sosstsst")
            missing_fields = [field for field in required_fields if field not in back_fields]
            if missing_fields:
                raise KeyError(
                    "assim_type='physical_ocean' requires fields "
                    f"{required_fields}; missing {missing_fields} in data_config['back_field']"
                )
            assimilate_restarts_physical_ocean(
                restart_dir,
                x[0][back_fields.index("siconc")],
                x[0][back_fields.index("sithic")],
                x[0][back_fields.index("sosstsst")],
                self.config['domain'],
                ocean_config=self.config.get("ocean_restart", {}),
            )
        elif self.config['assim_type'] == 'physical_ocean_bounded':
            back_fields = list(self.data_config.get("back_field", []))
            required_fields = ("siconc", "sithic", "sosstsst")
            missing_fields = [field for field in required_fields if field not in back_fields]
            if missing_fields:
                raise KeyError(
                    "assim_type='physical_ocean_bounded' requires fields "
                    f"{required_fields}; missing {missing_fields} in data_config['back_field']"
                )
            assimilate_restarts_physical_ocean_bounded(
                restart_dir,
                x[0][back_fields.index("siconc")],
                x[0][back_fields.index("sithic")],
                x[0][back_fields.index("sosstsst")],
                self.config['domain'],
                ocean_config=self.config.get("ocean_restart", {}),
            )
        elif self.config['assim_type'] == 'naively':
            assimilate_restarts_naively(restart_dir, x[0][0], x[0][1], self.config['domain'])
        else:
            raise KeyError(f"assim_type = {self.config['assim_type']} is not implemented")
