"""Resolve physical restart inputs without inventing unpredicted VAE fields."""

import numpy as np


BACKGROUND_INDICES = {'sithic': 1, 'sitemp': 2, 'sosstsst': 8}


def restart_inputs(analysis, fields, background, fallback_fields, required_fields):
    fields = list(fields)
    if len(fields) != len(analysis) or len(set(fields)) != len(fields):
        raise ValueError('Decoded fields must uniquely identify the analysis channels')
    resolved = dict(zip(fields, analysis))
    sources = {field: 'analysis' for field in fields}
    if not isinstance(fallback_fields, (list, tuple)):
        raise ValueError('restart_background_fields must be a list of missing fields')
    for field in fallback_fields:
        if field not in BACKGROUND_INDICES:
            raise ValueError(f'Unsupported background restart field: {field}')
        if field in resolved:
            raise ValueError(f'Background fallback would replace a decoded field: {field}')
        value = np.asarray(background[BACKGROUND_INDICES[field]])
        if value.shape != np.asarray(analysis[0]).shape:
            raise ValueError(f'Background and decoded grids differ for {field}')
        # InferenceDS.background is the original physical field at record 23.
        # No normalization, clipping or model-derived adjustment is applied.
        resolved[field] = value.copy()
        sources[field] = 'background'
    missing = [field for field in required_fields if field not in resolved]
    if missing:
        raise KeyError(f'Restart update requires fields {missing}; explicitly select background fallbacks')
    return resolved, sources
