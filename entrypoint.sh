#!/bin/sh
set -eu

python3 scripts/train_vae.py --config real_configs/training/vae_4f_um_multiscale_sobel_binary.yaml
python3 scripts/run_assimilation.py --config real_configs/assimilation/m2m/vae_4f_um_multiscale_sobel_binary_main_200d.yaml
python3 scripts/run_assimilation.py --config real_configs/assimilation/s2m/vae_4f_um_multiscale_sobel_binary_obs6_200d.yaml
python3 scripts/run_forecast_cycle.py --config real_configs/assimilation/restart/ice_ocean_bounded.yaml
