#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="${project_root}/config/experiments/s2m"
configs=(
  vae_1f_s2m_200d.yaml
  vae_1f_512_s2m_200d.yaml
  vae_3f_um_emb_s2m_200d.yaml
  vae_4f_um_s2m_200d.yaml
  vae_4f_um_sobel_binary_s2m_200d.yaml
  vae_4f_um_multiscale_s2m_200d.yaml
  vae_4f_um_multiscale_block_s2m_200d.yaml
  vae_4f_um_multiscale_sobel_binary_s2m_200d.yaml
  blue_s2m_200d.yaml
  var_100_s2m_200d.yaml
  enkf_correlated_s2m_200d.yaml
)
for config in "${configs[@]}"; do
  python "${project_root}/scripts/run_assimilation.py" \
    --config "${config_dir}/${config}" "$@"
done
