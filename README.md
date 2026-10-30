# VAE-MDA

Public code for **“Variational multivariate data assimilation improves sea-ice
analysis and forecasting.”** It includes VAE training, M2M and S2M
assimilation, NEMO/SI3 restart updates, baselines, and evaluation tools. Data,
weights, restarts, and outputs are not included.

## Quick start

~~~bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install clearml==1.13.1   # optional: enables online logging
cp .env.example .env                     # optional: add ClearML credentials
~~~

Runnable paper configs are in [real_configs/](real_configs/README.md). Paths
come directly from YAML. ClearML is an optional dependency and is discovered
automatically: if the `clearml` package is installed and the `CLEARML_*`
variables are set in `.env` (or a `~/clearml.conf` from `clearml-init` exists),
runs log to ClearML; otherwise they log locally to `outputs/`.

1. Edit YAML paths if they differ on your machine.
2. Edit [entrypoint.sh](entrypoint.sh) so exactly one command is uncommented.
3. Run:

~~~bash
./entrypoint.sh
~~~

For data requirements and validation, see [DATA.md](DATA.md). The larger
config/ tree contains editable templates, not ready-to-run configs.

## Paper experiments

The editable templates in `config/` (placeholder paths `/path/to/vae-mda-*`) cover the experiments of the paper:

| Experiment | Configs | Command |
|---|---|---|
| M2M and S2M analyses (Tables 1 and 2) | `config/experiments/m2m/`, `config/experiments/s2m/`: each VAE model at its selected loss weights (Supplementary Table S4), and the classical baselines | `scripts/run_m2m_200d.sh`, `scripts/run_s2m_200d.sh` |
| Loss-weight selection | `config/experiments/loss_weights.yaml` (60 development dates, weight grids) | `scripts/select_loss_weights.py --setting M2M` (or `S2M`) |
| Paired model comparisons with simultaneous intervals (Supplementary Table S7) | the `metrics.csv` of the runs above | `scripts/compare_models.py` |
| Background-term ablation | `config/experiments/multivariate/` (original loss weights) | `scripts/run_multivariate_penalty_200d.sh` |
| Restart-transfer comparison (Fig. 4a) | `config/restart/ice_only.yaml`, `config/restart/ice_ocean_bounded.yaml` | `scripts/run_forecast_cycle.py` |
| Selected-configuration forecasts (Table 3, Fig. 4b) | `config/experiments/forecast_2023/` (ten branches) | `scripts/run_forecast_cycle.py`; scoring and paired comparison: `scripts/compare_forecast_branches.py` |

Notes:

- The M2M data configs of the single- and three-field models and of the classical baselines use the rule-11
  track cells (`sral_indx: 11`), as in the paper; earlier versions of these configs used an outdated rule.
- In the forecast branches, fields that a model does not decode are taken from the branch background
  (`restart_background_fields`), new-ice layer energies use the SI3 expression (`ocean_restart.ice_energy_scheme:
  si3_native`), and `freeze_vae: true` keeps the VAE fixed while the latent variable is optimized.
- NEMO/SI3 itself is not included: `run_forecast_cycle.py` writes the analyses into the restarts, and the
  forecasts are run by the NEMO workflow.

## Container

The root Dockerfile is the only image definition. It installs the experiment
stack plus the optional ClearML client and runs entrypoint.sh. Compose passes
`.env` into the container, so ClearML credentials placed there are discovered
automatically — no `clearml.conf` mount is required. (Drop the `clearml` line
in the Dockerfile for a lean, local-logging-only image.)

~~~bash
cp .env.example .env
# Edit .env (mount roots + optional ClearML credentials) and entrypoint.sh.
docker compose build
docker compose run --rm vae-mda
~~~

Licensed under BSD-3-Clause.
