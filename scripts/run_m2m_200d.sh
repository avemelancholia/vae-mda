#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="${project_root}/config/experiments/m2m"
configs=(
  vae_1f_main_200d.yaml
  vae_1f_512_main_200d.yaml
  vae_3f_um_emb_main_200d.yaml
  vae_4f_um_main_200d.yaml
  vae_4f_um_sobel_binary_main_200d.yaml
  vae_4f_um_multiscale_main_200d.yaml
  vae_4f_um_multiscale_block_main_200d.yaml
  vae_4f_um_multiscale_sobel_binary_main_200d.yaml
  blue_main_200d.yaml
  var_100_main_200d.yaml
  enkf_correlated_main_200d.yaml
)
for config in "${configs[@]}"; do
  python "${project_root}/scripts/run_assimilation.py" \
    --config "${config_dir}/${config}" "$@"
done
