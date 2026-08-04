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
