"""Fail-fast checks for running retained configurations on real data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, validate_references


class PreflightError(RuntimeError):
    """Raised when a real-data experiment is not ready to start."""


def workflow_name(config: dict[str, Any]) -> str:
    if "restart_dir" in config:
        return "restart assimilation"
    if "epoch" in config and config.get("model_name") == "diffusers":
        return "VAE training"
    return "offline assimilation"


def require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise PreflightError(f"Missing {label}: {path}")
    return path


def require_dir(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise PreflightError(f"Missing {label}: {path}")
    return path


def validate_checkpoint(model_config: dict[str, Any]) -> Path:
    checkpoint = require_dir(model_config["path"], "VAE checkpoint directory")
    if not (checkpoint / "config.json").is_file() and not (checkpoint / "model.pt").is_file():
        raise PreflightError(
            f"Checkpoint has neither config.json nor model.pt: {checkpoint}"
        )
    weight_files = [
        path
        for pattern in ("*.safetensors", "*.bin", "*.pt")
        for path in checkpoint.glob(pattern)
    ]
    if not weight_files:
        raise PreflightError(f"No checkpoint weights found in {checkpoint}")
    return checkpoint


def validate_static_model_inputs(model_config: dict[str, Any]) -> list[Path]:
    checked = []
    for key, value in model_config.items():
        if key.endswith("_path") and key != "path" and value:
            checked.append(require_file(value, key))
    return checked


def array_shape(path: Path) -> tuple[int, ...]:
    try:
        return tuple(np.load(path, mmap_mode="r").shape)
    except Exception as exc:
        raise PreflightError(f"Cannot read NumPy array {path}: {exc}") from exc


def validate_prediction_shape(path: Path, data_config: dict[str, Any], mask_shape) -> tuple[int, ...]:
    shape = array_shape(path)
    indices = [int(index) for index in data_config["indxs"]]
    if len(shape) != 4:
        raise PreflightError(
            f"NEMO prediction must have [variable, hour, y, x] shape: {path} has {shape}"
        )
    if shape[0] <= max(indices) or shape[1] < 24:
        raise PreflightError(
            f"NEMO prediction lacks required variables/hours: {path} has {shape}, "
            f"indices={indices}"
        )
    if tuple(shape[-2:]) != tuple(mask_shape):
        raise PreflightError(
            f"Prediction grid {shape[-2:]} does not match land mask {mask_shape}: {path}"
        )
    return shape


def _dataset_sample_shape(sample) -> str:
    def describe(value):
        if hasattr(value, "shape"):
            return tuple(value.shape)
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        return type(value).__name__

    return repr(describe(sample))


def _finite_count(value) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return int(np.isfinite(np.asarray(value)).sum())


def _require_finite(value, label: str) -> int:
    count = _finite_count(value)
    if count == 0:
        raise PreflightError(f"Loaded sample has no finite {label}")
    return count


def _has_paths(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return bool(value)
    return bool(value)


def check_training(config, data_config, model_config, *, load_sample: bool) -> list[str]:
    from vae_mda.data.vae_data import VAETrainDS

    mask_path = require_file(data_config["mask_path"], "land mask")
    mask_shape = array_shape(mask_path)
    preds_dir = require_dir(Path(data_config["dataset_dir"]) / "preds", "NEMO predictions")
    train_dataset = VAETrainDS(data_config.copy(), tag="train")
    val_dataset = VAETrainDS(data_config.copy(), tag="val")
    first_prediction = Path(train_dataset.data.iloc[0].pred)
    prediction_shape = validate_prediction_shape(first_prediction, data_config, mask_shape)
    messages = [
        f"train samples: {len(train_dataset)}",
        f"validation samples: {len(val_dataset)}",
        f"prediction shape: {prediction_shape}",
        f"model output directory: {model_config['path']}",
    ]
    if load_sample:
        messages.append(f"loaded training sample: {_dataset_sample_shape(train_dataset[0])}")
        messages.append(f"loaded validation sample: {_dataset_sample_shape(val_dataset[0])}")
    return messages


def check_offline(config, data_config, model_config, *, load_sample: bool) -> list[str]:
    from vae_mda.data.assimilation_data import (
        Fields3DVARDS,
        Fields3DVAREXPDS,
        Fields3DVARSampledDS,
    )

    datasets = {
        cls.name: cls for cls in (Fields3DVARDS, Fields3DVAREXPDS, Fields3DVARSampledDS)
    }
    try:
        dataset_class = datasets[data_config["dataset_name"]]
    except KeyError as exc:
        raise PreflightError(f"Unsupported dataset: {exc.args[0]}") from exc

    mask_path = require_file(data_config["mask_path"], "land mask")
    mask_shape = array_shape(mask_path)
    require_dir(Path(data_config["dataset_dir"]) / "preds", "NEMO predictions")
    require_dir(data_config["sral_dir"], "SRAL inputs")
    amsr_dir = data_config.get("amsr_dir", Path(data_config["sat_dir"]) / "sats/amsr2")
    require_dir(amsr_dir, "AMSR2 inputs")

    dataset = dataset_class(data_config.copy())
    if len(dataset) == 0:
        raise PreflightError("Dataset contains no assimilation cases")
    base_dataset = dataset
    if hasattr(dataset, "sample_indices"):
        first_index = dataset.sample_indices[0]
    else:
        first_index = 0
    first_prediction = Path(base_dataset.back_data.iloc[first_index].pred)
    prediction_shape = validate_prediction_shape(first_prediction, data_config, mask_shape)

    missing_observations = []
    missing_next_validation = []
    missing_amsr = []
    for index in range(len(dataset.back_data)):
        obs_index = index + dataset.obs_shift
        observation_window = dataset.obs_data.iloc[
            obs_index - dataset.obs_shift : obs_index + 1
        ]
        if not any(_has_paths(value) for value in observation_window.sral):
            missing_observations.append(str(dataset.back_data.iloc[index].date.date()))
        if not _has_paths(dataset.obs_data.iloc[obs_index + 1].sral):
            missing_next_validation.append(str(dataset.back_data.iloc[index].date.date()))
        if data_config["ass_data"] == "sral" and not _has_paths(
            dataset.obs_data.iloc[obs_index].amsr
        ):
            missing_amsr.append(str(dataset.back_data.iloc[index].date.date()))
    if missing_observations:
        raise PreflightError(
            f"No usable SRAL observation window for {len(missing_observations)} cases; "
            f"first dates: {missing_observations[:5]}"
        )
    if missing_next_validation:
        raise PreflightError(
            f"Missing next-day SRAL validation for {len(missing_next_validation)} cases; "
            f"first dates: {missing_next_validation[:5]}"
        )
    if missing_amsr:
        raise PreflightError(
            f"Missing same-day AMSR2 validation for {len(missing_amsr)} cases; "
            f"first dates: {missing_amsr[:5]}"
        )

    checked_model_files = validate_static_model_inputs(model_config)
    messages = [
        f"assimilation cases: {len(dataset)}",
        f"prediction shape: {prediction_shape}",
        "SRAL/validation coverage: complete",
    ]
    if config["model_name"].startswith("vae_"):
        messages.append(f"checkpoint: {validate_checkpoint(model_config)}")
    messages.extend(f"model input: {path}" for path in checked_model_files)
    if load_sample:
        sample = dataset[0]
        tensors, val_1, val_2, _ = sample
        background_count = _require_finite(tensors[0], "background values")
        observation_count = _require_finite(tensors[2], "observation values")
        validation_1_count = _require_finite(val_1, "same-day validation values")
        validation_2_count = _require_finite(val_2, "next-day validation values")
        messages.append(f"loaded assimilation sample: {_dataset_sample_shape(sample)}")
        messages.append(
            "finite values: "
            f"background={background_count}, observations={observation_count}, "
            f"validation={validation_1_count}/{validation_2_count}"
        )
    return messages


def check_restart(config, data_config, model_config, *, load_sample: bool, run_date: str | None) -> list[str]:
    from vae_mda.data.inference_data import InferenceDS

    selected_date = date.fromisoformat(run_date or config["date"])
    mask_path = require_file(data_config["mask_path"], "land mask")
    mask_shape = array_shape(mask_path)
    require_file(config["domain"], "NEMO domain file")
    checkpoint = validate_checkpoint(model_config)
    sral_dir = require_dir(config["sral_dir"], "SRAL inputs")

    run_dir = require_dir(
        Path(config["restart_dir"]) / f"run_{selected_date.isoformat()}T00:00:00+00:00",
        "forecast run directory",
    )
    require_file(run_dir / "forecast-done", "forecast completion marker")
    for suffix in (
        "_forecast.1h_icemod.nc",
        "_forecast.1h_gridVsurf.nc",
        "_forecast.1h_gridTsurf.nc",
    ):
        if not any(path.name.endswith(suffix) for path in run_dir.iterdir()):
            raise PreflightError(f"No {suffix} file in {run_dir}")

    restart_dir = require_dir(run_dir / "nemo_restart", "NEMO restart directory")
    names = [path.name for path in restart_dir.iterdir() if path.is_file()]
    if not any("restart_ice" in name for name in names):
        raise PreflightError(f"No restart_ice files in {restart_dir}")
    if config["assim_type"] in {"physical_ocean", "physical_ocean_bounded"}:
        ocean = [
            name for name in names
            if "restart_ice" not in name
            and name.endswith(".nc")
            and (name.startswith("restart-") or name.startswith("restart_") or name == "restart.nc")
        ]
        if not ocean:
            raise PreflightError(f"No ocean restart files in {restart_dir}")

    missing_sral = []
    for offset in range(int(data_config.get("assimilation_range", 1))):
        day = selected_date - timedelta(days=offset)
        path = sral_dir / f"{day.isoformat()}.npy"
        if not path.is_file():
            missing_sral.append(path)
    if missing_sral:
        raise PreflightError(
            "Missing restart SRAL inputs: " + ", ".join(str(path) for path in missing_sral)
        )

    messages = [
        f"date: {selected_date.isoformat()}",
        f"land-mask shape: {mask_shape}",
        f"checkpoint: {checkpoint}",
        f"forecast directory: {run_dir}",
        f"restart directory: {restart_dir}",
    ]
    if load_sample:
        sample_config = config.copy()
        sample_config["date"] = selected_date.isoformat()
        dataset = InferenceDS(sample_config, data_config.copy())
        sample = dataset[0]
        tensors, _ = sample
        background_count = _require_finite(tensors[0], "restart background values")
        observation_count = _require_finite(tensors[2], "restart observation values")
        messages.append(f"loaded restart sample: {_dataset_sample_shape(sample)}")
        messages.append(
            f"finite values: background={background_count}, observations={observation_count}"
        )
    return messages


def run_preflight(config_path: str | Path, *, load_sample: bool = False, run_date: str | None = None) -> list[str]:
    config = load_config(config_path)
    validate_references(config)
    data_config = load_config(config["data_config"])
    model_config = load_config(config["model_config"])
    workflow = workflow_name(config)
    messages = [f"workflow: {workflow}", f"config: {config['_config_path']}"]
    if workflow == "VAE training":
        messages.extend(check_training(config, data_config, model_config, load_sample=load_sample))
    elif workflow == "offline assimilation":
        messages.extend(check_offline(config, data_config, model_config, load_sample=load_sample))
    else:
        messages.extend(
            check_restart(
                config,
                data_config,
                model_config,
                load_sample=load_sample,
                run_date=run_date,
            )
        )
    return messages
