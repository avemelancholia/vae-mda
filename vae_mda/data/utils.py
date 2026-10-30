import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr


from pathlib import Path
from scipy.interpolate import griddata

"""
    [0]  'siconc'       - концентрация льда,
    [1]  'sithic'       - толщина льда,
    [2]  'sitemp'       - темпретура льда,
    [3]  'snthic'       - толщина снега,
    [4]  'sivelu'       - скорость льда х-комп (в проекции на сетку модели),
    [5]  'sivelv'       - скорость льда у-комп (в проекции на сетку модели),
    [6]  'vozocrtx'     - скорость течения на поверхности (в проекции на сетку модели),
    [7]  'vomecrty'     - скорость течения на поверхности (в проекции на сетку модели),
    [8] 'sosstsst'      - температура воды на поверхности,
    [9] 'sosaline'      - соленость воды на поверхности,
    [10] 'T2'           - температура на высоте двух метров
    [11] 'SWDOWN'       - коротковолновое излучение,
    [12] 'GLW'          - длинноволновое излучение,
    [13]  'U10'         - скорость ветра на высоте 10м х-комп (в проекции на сетку модели),
    [14]  'V10'         - скорость ветра на высоте 10м у-комп (в проекции на сетку модели),
"""

nemo_folder = {
    24: '/ocean+atmosphere_24',
    48: '/ocean+atmosphere_48',
    72: '/ocean+atmosphere_72',
    'all': '',
}

# ---------------
# data from .nc param
class LoadStructure:
    def __init__(self):
        self.vars_icemod = ['siconc', 'sithic', 'sitemp', 'snthic']
        self.flow_icemod =[['sivelu', 'sivelv'], ]

        self.vars_gridvsurf = [] 
        self.vars_gridvsurf_flow = [['vozocrtx', 'vomecrty'], ]

        self.vars_gridtsurf = ['sosstsst', 'sosaline']

        self.vars_wrf = ['T2', 'SWDOWN', 'GLW'] 
        self.vars_wrf_flow = [['U10', 'V10'],]


def make_sobel(array):
    assert len(array.shape)==3
    # Применение фильтра Собеля
    sobelx = ndimage.sobel(array, axis=1)  # x-производная
    sobely = ndimage.sobel(array, axis=2)  # y-производная

    # Вычисление величины градиента
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return gradient_magnitude

def interp_wrf_to_nemo(var, shape, wrf_coords, nemo_coords):
    new_var = griddata(wrf_coords, var, nemo_coords, method='linear')
    new_var = new_var.reshape(shape)
    return new_var

def make_sym_border(array):
    array[0, :] = array[1, :]
    array[-1, :] = array[-2, :]
    array[:, 0] = array[:, 1]
    array[:, -1] = array[:, -2]
    return array

def invert_matrix_vectorized(M):
    det_M = M[:, :, 0, 0] * M[:, :, 1, 1] - M[:, :, 0, 1] * M[:, :, 1, 0]
    alg_M = np.zeros_like(M)
    alg_M[:, :, 0, 0] = M[:, :, 1, 1]
    alg_M[:, :, 1, 1] = M[:, :, 0, 0]
    alg_M[:, :, 1, 0] = -M[:, :, 0, 1]
    alg_M[:, :, 0, 1] = -M[:, :, 1, 0]
    return np.transpose(alg_M, (0, 1, 3, 2)) / np.expand_dims(det_M, (2, 3))

def init_coefs(lats, lons):
    lats_x = np.zeros_like(lats)
    lats_y = np.zeros_like(lats)
    lons_x = np.zeros_like(lats)
    lons_y = np.zeros_like(lats)

    lats_x[:-1, :] = lats[1:, :] - lats[:-1, :]
    lats_y[:, :-1] = lats[:, 1:] - lats[:, :-1]
    lons_x[:-1, :] = lons[1:, :] - lons[:-1, :]
    lons_y[:, :-1] = lons[:, 1:] - lons[:, :-1]

    lats_x = make_sym_border(lats_x)
    lats_y = make_sym_border(lats_y)
    lons_x = make_sym_border(lons_x)
    lons_y = make_sym_border(lons_y)

    A = np.zeros(lats.shape + (2, 2))

    dists_lat = 111.6 * 1000 * np.ones_like(lats)
    dists_lon = np.cos(lats * np.pi / 180.0) * 111.6 * 1000
    
    dists_x = np.sqrt(np.square(lats_x*dists_lat) + np.square(lons_x*dists_lon))
    dists_y = np.sqrt(np.square(lats_y*dists_lat) + np.square(lons_y*dists_lon))
    
    A[:, :, 0, 0] = lats_x*dists_lat/dists_x
    A[:, :, 0, 1] = lats_y*dists_lat/dists_y
    A[:, :, 1, 0] = lons_x*dists_lon/dists_x
    A[:, :, 1, 1] = lons_y*dists_lon/dists_y

    B = A
    B = invert_matrix_vectorized(A)
    
    return B, dists_y, dists_x

def calculate_ice_flow(curr_v, curr_u, B, dists_y, dists_x):
    curr_v_normed = curr_v
    curr_u_normed = curr_u
    curr_stacked = np.stack((curr_v_normed, curr_u_normed), 2)
    curr_stacked = np.expand_dims(curr_stacked, 2)
    
    print(curr_stacked.shape)

    flow = curr_stacked @ B
    flow[:, :, 0, 0] = 3600*flow[:, :, 0, 0]/dists_x
    flow[:, :, 0, 1] = 3600*flow[:, :, 0, 1]/dists_y
    return flow

def move_by_flow(x, flow, unsqueeze=True):
    x_mesh, y_mesh = np.meshgrid(np.arange(225), np.arange(311))
    x_mesh = torch.tensor(x_mesh, dtype=x.dtype, device=x.device)
    y_mesh = torch.tensor(y_mesh, dtype=x.dtype, device=x.device)

    x_mesh = torch.cat(
        [torch.unsqueeze(x_mesh, dim=0) for i in range(x.shape[0])]
    )
    y_mesh = torch.cat(
        [torch.unsqueeze(y_mesh, dim=0) for i in range(x.shape[0])]
    )  
    flow = flow[:, :, :, 0, :]
    
    flow[..., 0] = x_mesh.to(x) + flow[..., 0]
    flow[..., 1] = y_mesh.to(x) + flow[..., 1]

    flow[..., 0] = (flow[..., 0] - 224.0 / 2) / 224.0 * 2
    flow[..., 1] = (flow[..., 1] - 310.0 / 2) / 310.0 * 2

    if unsqueeze:
        x = x.unsqueeze(dim=1)
    pred = torch.nn.functional.grid_sample(
        x, flow.to(x), align_corners=True
    )
    if unsqueeze:
        return pred[:, 0]
    else:
        return pred

def is_nemo_ready(nemo_dir, date):
    date_str = f'run_{date.strftime("%Y-%m-%d")}T00:00:00+00:00'
    path = nemo_dir / date_str
    print(path)
    if not path.exists():
        return False

    files = os.listdir(path)

    icemod = [f for f in files if f.endswith('icemod.nc')]
    gridvsurf = [f for f in files if f.endswith('gridVsurf.nc')]
    gridtsurf = [f for f in files if f.endswith('gridTsurf.nc')]

    if len(icemod) == 0 or len(gridvsurf) == 0 or len(gridtsurf) == 0:
        return False
    else:
        return True

def is_wrf_ready(wrf_dir, date):
    name_str = f'wrfout_d01_{date.strftime("%Y-%m-%d")}_00:00:00'
    if (wrf_dir / name_str).exists():
        return True
    return False

class FlowL():
    def __init__(self, lats, lons):
        self.init_coefs(lats, lons)
        
    def init_coefs(self, lats, lons):
        lats_x = np.zeros_like(lats)
        lats_y = np.zeros_like(lats)
        lons_x = np.zeros_like(lats)
        lons_y = np.zeros_like(lats)

        lats_x[:-1, :] = lats[1:, :] - lats[:-1, :]
        lats_y[:, :-1] = lats[:, 1:] - lats[:, :-1]
        lons_x[:-1, :] = lons[1:, :] - lons[:-1, :]
        lons_y[:, :-1] = lons[:, 1:] - lons[:, :-1]

        lats_x = make_sym_border(lats_x)
        lats_y = make_sym_border(lats_y)
        lons_x = make_sym_border(lons_x)
        lons_y = make_sym_border(lons_y)

        A = np.zeros(lats.shape + (2, 2))

        dists_lat = 111.6 * 1000 * np.ones_like(lats)
        dists_lon = np.cos(lats * np.pi / 180.0) * 111.6 * 1000
    
        self.dists_x = np.sqrt(np.square(lats_x*dists_lat) + np.square(lons_x*dists_lon))
        self.dists_y = np.sqrt(np.square(lats_y*dists_lat) + np.square(lons_y*dists_lon))
    
        A[:, :, 0, 0] = lats_x*dists_lat/self.dists_x
        A[:, :, 0, 1] = lons_x*dists_lon/self.dists_x
        A[:, :, 1, 0] = lats_y*dists_lat/self.dists_y
        A[:, :, 1, 1] = lons_y*dists_lon/self.dists_y

        self.B = A

    def calculate_ice_flow(self, curr_x, curr_y):
        curr_x_normed = curr_x
        curr_y_normed = curr_y
        curr_stacked = np.stack((curr_x_normed, curr_y_normed), 2)
        curr_stacked = np.expand_dims(curr_stacked, 2)

        flow = curr_stacked @ self.B
        flow[:, :, 0, 0] = 3600*flow[:, :, 0, 0]/self.dists_x
        flow[:, :, 0, 1] = 3600*flow[:, :, 0, 1]/self.dists_y
        return -flow

def calculate_flows_l(lats, lons):
    flow_class = FlowL(lats, lons)
    def calculate(us, vs):
        xs, ys = [], []
        for i in range(us.shape[0]):
            flow = flow_class.calculate_ice_flow(us[i], vs[i])
            x, y = flow[:, :, 0, 0], flow[:, :, 0, 1]
            xs.append(x[None, ...])
            ys.append(y[None, ...])
        return np.concatenate(xs), np.concatenate(ys)
    return calculate

def get_nc_data(nemo_dir, wrf_dir, date):
    hour_start = 0
    nemo_dir = Path(nemo_dir)
    wrf_dir = Path(wrf_dir)
    structure = LoadStructure()
    
    data = []
    if is_nemo_ready(nemo_dir, date):
        nemo_path = nemo_dir / \
            f'run_{date.strftime("%Y-%m-%d")}T00:00:00+00:00'
        nemo_files = os.listdir(nemo_path)
        icemod = [f for f in nemo_files if f.endswith('_forecast.1h_icemod.nc')][0]
        gridvsurf = [f for f in nemo_files if f.endswith('_forecast.1h_gridVsurf.nc')][0]
        gridtsurf = [f for f in nemo_files if f.endswith('_forecast.1h_gridTsurf.nc')][0]

        icemod = nemo_path / icemod
        gridvsurf = nemo_path / gridvsurf
        gridtsurf = nemo_path / gridtsurf

        ds = xr.open_dataset(icemod)
        lats = ds.nav_lat.values
        lons = ds.nav_lon.values
        calculate_flows = calculate_flows_l(lats, lons)
        nemo_coords = np.stack((
            ds['nav_lon'].data.ravel(),
            ds['nav_lat'].data.ravel(),
        ), axis=1)
        
        for var in structure.vars_icemod:
            field = ds[var].data[hour_start: hour_start+24]
            data.append(field.astype(np.float16))
        for var_u, var_v in structure.flow_icemod:
            us = ds[var_u].data[hour_start: hour_start+24]
            vs = ds[var_v ].data[hour_start: hour_start+24]
            var_x, var_y = calculate_flows(us, vs)
            data.append(var_x.astype(np.float16))
            data.append(var_y.astype(np.float16))
        ds.close()

        ds = xr.open_dataset(gridvsurf)
        for var in structure.vars_gridvsurf:
            data.append(ds[var].data[hour_start: hour_start+24].astype(np.float32))
        for var_u, var_v in structure.vars_gridvsurf_flow:
            us = ds[var_u].data[hour_start: hour_start+24]
            vs = ds[var_v ].data[hour_start: hour_start+24]
            var_x, var_y = calculate_flows(us, vs)
            data.append(var_x.astype(np.float16))
            data.append(var_y.astype(np.float16))
        ds.close()
        
        ds = xr.open_dataset(gridtsurf)
        for var in structure.vars_gridtsurf:
            data.append(ds[var].data[hour_start: hour_start+24].astype(np.float32))
        ds.close()
    
    else:
        raise FileNotFoundError(f"Error: Nemo files are not found, chek: {nemo_dir}")
    wrf_exist = is_wrf_ready(wrf_dir, date)

    if wrf_exist:
        date_wrf = date.add(hours=hour_start)
        wrf_file = wrf_dir / f'wrfout_d01_{date_wrf.strftime("%Y-%m-%d")}_00:00:00'
        ds = xr.open_dataset(wrf_file)
        wrf_coords = np.stack((
            ds['XLONG'].data[0].ravel(),
            ds['XLAT'].data[0].ravel()
        ), axis=1)
        if len(ds['U10']) < 24:
            warnings.warn(f"WRF is too short: len wrf {len(ds['U10'])} < 24")
            ds.close()
            wrf_exist = False
    else:
        warnings.warn(f"WRF file is not found: check {wrf_dir}")

    for var in structure.vars_wrf:
        arrays = []
        for i in range(24):
            if wrf_exist: 
                array = ds[var].data[i].astype(np.float32)
                array = interp_wrf_to_nemo(array.ravel(), (311, 225), wrf_coords, nemo_coords)
            else: 
                array = np.full((311, 225), np.nan)
            arrays.append(array)
        arrays = np.stack(arrays, axis=0)
        data.append(arrays)

    for var_u, var_v in structure.vars_wrf_flow:
        us, vs = [], []
        for i in range(24):
            if wrf_exist:
                u = ds[var_u].data[i].astype(np.float32)
                u = interp_wrf_to_nemo(u.ravel(), (311, 225), wrf_coords, nemo_coords)
            else:
                u = np.full((311, 225), np.nan)
            us.append(u)
            if wrf_exist:
                v = ds[var_v].data[i].astype(np.float32)
                v = interp_wrf_to_nemo(v.ravel(), (311, 225), wrf_coords, nemo_coords)
            else:
                v = np.full((311, 225), np.nan)   
            vs.append(v)
        us, vs = np.stack(us, axis=0), np.stack(vs, axis=0)
        var_x, var_y = calculate_flows(us, vs)
        data.append(var_x.astype(np.float16))
        data.append(var_y.astype(np.float16))

    if wrf_exist:
        ds.close()

    data = np.stack(data, axis=0)
    data = data.astype(np.float16)

    return data
# ---------------

def fill_mask(x, mask, filler=np.nan):
    x_mask = np.isnan(x)
    x[np.logical_and(~mask, x_mask)] = filler
    x[mask] = np.nan
    return x

def get_df(preds_dir, sral_dir=None, amsr_dir=None):
    """Build the dated input index used by training and assimilation datasets."""

    preds_dir = Path(preds_dir)
    if not preds_dir.is_dir():
        raise FileNotFoundError(f"NEMO prediction directory not found: {preds_dir}")

    pred_paths = sorted(path for path in preds_dir.iterdir() if path.suffix == ".npy")
    if not pred_paths:
        raise FileNotFoundError(f"No .npy prediction files found in {preds_dir}")

    dates = []
    for pred_path in pred_paths:
        date_text = pred_path.stem.rsplit("_", 1)[-1]
        try:
            dates.append(pd.to_datetime(date_text, format="%Y-%m-%d"))
        except ValueError as exc:
            raise ValueError(
                f"Prediction filename must end in _YYYY-MM-DD.npy: {pred_path.name}"
            ) from exc

    payload = {
        "date": dates,
        "pred": [str(path) for path in pred_paths],
    }

    if sral_dir is not None or amsr_dir is not None:
        if sral_dir is None or amsr_dir is None:
            raise ValueError("sral_dir and amsr_dir must be provided together")
        sral_dir = Path(sral_dir)
        amsr_dir = Path(amsr_dir)
        if not sral_dir.is_dir():
            raise FileNotFoundError(f"SRAL directory not found: {sral_dir}")
        if not amsr_dir.is_dir():
            raise FileNotFoundError(f"AMSR2 directory not found: {amsr_dir}")

        sral_paths = sorted(path for path in sral_dir.iterdir() if path.suffix == ".npy")
        payload["sral"] = [
            [str(path) for path in sral_paths if date.strftime("%Y-%m-%d") in path.stem]
            or None
            for date in dates
        ]
        payload["amsr"] = []
        for date in dates:
            amsr_path = amsr_dir / f"asi-AMSR2-n3125-{date.strftime('%Y%m%d')}.npy"
            payload["amsr"].append([str(amsr_path)] if amsr_path.is_file() else None)

    return pd.DataFrame.from_dict(payload)
