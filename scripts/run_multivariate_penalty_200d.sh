#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="${project_root}/config/experiments/multivariate"
for config in "${config_dir}"/*.yaml; do
  python "${project_root}/scripts/run_assimilation.py" --config "${config}" "$@"
done
