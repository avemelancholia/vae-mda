# Data layout

Runtime paths are defined in YAML. Research data and weights are not included.

~~~text
/mnt/sciml/data_assimilation/
├── da_arctic_2015-2024_v0.1/preds/
├── da_arctic_v0.4/
├── sral_si/sral_new_format/
├── amsr2/
├── land_mask.npy
├── weights/vae_4f_um_multiscale_sobel_binary/
└── paper_vae_borey/<restart directory>/

/mnt/nestp/temp/BARKA12_domain_cfg_v1.3.nc
~~~

Requirements:

- Prediction files end in _YYYY-MM-DD.npy and have shape
  [variable, forecast_hour, y, x] with at least 24 hours.
- Model, mask, and satellite arrays use the same horizontal grid.
- Concentrations are fractions in [0, 1].
- Restart runs contain forecast-done and nemo_restart/.

Validate one production sample with:

~~~bash
python scripts/preflight.py --config <experiment.yaml> --load-sample
~~~
