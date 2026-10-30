import xarray as xr
import numpy as np
import pendulum as pdl
import argparse
import os

from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LinearRegression
from scipy.signal import convolve2d

from matplotlib import pyplot as plt
from pathlib import Path
import os

from .si3_energy import new_ice_energy_density


def extract_coords(pieces):
    out = {}
    for piece in pieces:
        pdict = {}
        with xr.open_dataset(piece) as f:
            pdict["nav_lat"] = f["nav_lat"].data
            pdict["nav_lon"] = f["nav_lon"].data
        out[piece] = pdict
    return out


def map_piece(piece, domain):
    def extract_common_pair(pair_1, pair_2):
        pair_1 = set([tuple(i) for i in pair_1])
        pair_2 = set([tuple(i) for i in pair_2])
        return list(pair_1.intersection(pair_2))[0]

    mapping = {}
    lat_00 = piece["nav_lat"][0, 0]
    lon_00 = piece["nav_lon"][0, 0]
    lat_11 = piece["nav_lat"][-1, -1]
    lon_11 = piece["nav_lon"][-1, -1]

    dom_lat = domain["nav_lat"]
    dom_lon = domain["nav_lon"]

    mapping["ind_00"] = extract_common_pair(
        np.argwhere(dom_lat == lat_00), np.argwhere(dom_lon == lon_00)
    )
    mapping["ind_11"] = extract_common_pair(
        np.argwhere(dom_lat == lat_11), np.argwhere(dom_lon == lon_11)
    )
    return mapping


def check_coverage(mapping, coords_domain):
    domain = np.ones_like(coords_domain["nav_lat"])
    area = np.sum(domain)
    for piece, coords in mapping.items():
        x0, y0 = coords["ind_00"]
        x1, y1 = coords["ind_11"]
        domain[x0 : x1 + 1, y0 : y1 + 1] = 0
    coverage_fraction = 1 - np.sum(domain) / area
    print(f"Restart tile coverage of full grid (including land): {100 * coverage_fraction:.2f}%")


def create_mapping(pieces, domain):
    mapping = {}

    pieces = sorted(pieces)

    coords_pieces = extract_coords(pieces)
    with xr.open_dataset(domain) as f:
        coords_domain = {"nav_lat": f["nav_lat"].data, "nav_lon": f["nav_lon"].data}
    for piece, coords in coords_pieces.items():
        name = f"index_{str(piece).split('_')[-1][:-3]}"
        mapping[name] = map_piece(coords, coords_domain)
    check_coverage(mapping, coords_domain)

    mapping["size_output"] = coords_domain["nav_lat"].shape

    return mapping


def map_pieces(mappable, mapping):
    if isinstance(mappable, dict):
        mappable_is_pieces = True
    else:
        mappable_is_pieces = False
    if mappable_is_pieces:
        sample_piece = list(mappable.values())[0]
        output_shape = sample_piece.shape[:-2] + mapping["size_output"]
        output = np.zeros(output_shape)
        for piece, array in mappable.items():
            # name = f"index_{str(piece).split('_')[-1][:-3]}"
            piece_map = mapping[piece]
            output[
                ...,
                piece_map["ind_00"][0] : piece_map["ind_11"][0] + 1,
                piece_map["ind_00"][1] : piece_map["ind_11"][1] + 1,
            ] = array
    else:
        output = {}
        for name, indexes in mapping.items():
            if name.find("size") > -1:
                continue
            ind_00 = indexes["ind_00"]
            ind_11 = indexes["ind_11"]
            array = np.copy(
                mappable[..., ind_00[0] : ind_11[0] + 1, ind_00[1] : ind_11[1] + 1]
            )
            output[name] = array
    return output


def load_data(pieces, mapping, keys=None):
    data = {}
    output = {}
    valid_keys = []
    for piece in pieces:  # чтобы открыть один раз
        name = f"index_{str(piece).split('_')[-1][:-3]}"
        with xr.open_dataset(piece) as ds:
            if keys is None:
                keys = ds.variables

            for key in keys:
                if key not in data:
                    data[key] = {}
                shape = ds[key].data.shape
                name = f"index_{str(piece).split('_')[-1][:-3]}"

                if len(shape) == 3:
                    data[key][name] = ds[key].data[0] # add
                elif len(shape) == 4:
                    data[key][name] = ds[key].data[0][0] # add
                else:
                    continue
    for key in data.keys():
        if len(data[key]) > 0:
            output[key] = map_pieces(data[key], mapping)
    return output


def ed_nn_2d(img, mask_1, mask_2):
    """
    Заполняет пиксели изображения на основе двух масок.
    Явно использует евкидово расстояние.

    Параметры:
        img: np.array формы (height, width, ch) - исходное изображение
        mask_1: np.array формы (height, width) - первая булева маска
        mask_2: np.array формы (height, width) - вторая булева маска

    Возвращает:
        Обработанное изображение той же формы
    """
    # Создаем копию изображения для модификации
    result = img.copy()
    height, width, ch = img.shape

    # Случай 1: mask_1=True и mask_2=False
    # Заполняем ближайшими пикселями где mask_1=False
    zero_mask = img[:, :, 0] == 0  # это суша
    condition1 = mask_1 & ~mask_2
    if np.any(condition1):
        # Создаем маску источников (где брать значения)
        sources1 = ~mask_1 & ~zero_mask
        for c in range(ch):
            # Получаем индексы ближайших пикселей
            _, indices = distance_transform_edt(~sources1, return_indices=True)
            # Заполняем пиксели
            result[condition1, c] = img[
                indices[0][condition1], indices[1][condition1], c
            ]

    # Случай 2: mask_1=False и mask_2=True
    # Заполняем ближайшими пикселями где mask_1=True
    condition2 = ~mask_1 & mask_2
    if np.any(condition2):
        # Создаем маску источников (где брать значения)
        zero_mask = img[:, :, 0] == 0  # это суша
        sources2 = mask_1 & ~zero_mask
        for c in range(ch):
            # val = img[sources2, c].mean()
            # Получаем индексы ближайших пикселей
            _, indices = distance_transform_edt(~sources2, return_indices=True)
            # Заполняем пиксели
            # result[condition2, c] = val
            result[condition2, c] = img[
                indices[0][condition2], indices[1][condition2], c
            ]
    return result


def ed_nn_3d(img, mask_1, mask_2, depth):
    """
    Заполняет пиксели изображения на основе двух масок.
    Явно использует евкидово расстояние.

    Параметры:
        img: np.array формы (depth, height, width) - исходное изображение
        mask_1: np.array формы (height, width) - первая булева маска
        mask_2: np.array формы (height, width) - вторая булева маска

    Возвращает:
        Обработанное изображение той же формы
    """
    # Создаем копию изображения для модификации
    result = img.copy()
    _, height, width = img.shape

    # Случай 1: mask_1=True и mask_2=False
    # Заполняем ближайшими пикселями где mask_1=False
    condition1 = mask_1 & ~mask_2
    if np.any(condition1):
        # Создаем маску источников (где брать значения)
        for d in range(depth):
            zero_mask = img[d] == 0  # это суша
            sources1 = ~mask_1 & ~zero_mask
            to1 = condition1 & ~zero_mask
            val = img[d, sources1].mean()
            # Получаем индексы ближайших пикселей
            _, indices = distance_transform_edt(~sources1, return_indices=True)
            # Заполняем пиксели
            # тут делала средним
            # result[d, to1] = val*(depth-d)/depth + result[d, to1]*d/depth
            result[d, to1] = (
                img[d, indices[0][to1], indices[1][to1]] * (depth - d) / depth
                + result[d, to1] * d / depth
            )

    # Случай 2: mask_1=False и mask_2=True
    # Заполняем ближайшими пикселями где mask_1=True
    condition2 = ~mask_1 & mask_2
    if np.any(condition2):
        # Создаем маску источников (где брать значения)
        for d in range(depth):
            zero_mask = img[d] == 0  # это суша

            sources2 = mask_1 & ~zero_mask
            to1 = condition2 & ~zero_mask
            val = img[d, sources2].mean()
            # Получаем индексы ближайших пикселей
            _, indices = distance_transform_edt(~sources2, return_indices=True)
            # Заполняем пиксели

            # result[d, to1] = val*(depth-d)/depth + result[d, to1]*d/depth
            result[d, to1] = (
                img[d, indices[0][to1], indices[1][to1]] * (depth - d) / depth
                + result[d, to1] * d / depth
            )

    return result


def ed_nn_1(img, mask_1, mask_2):
    """
    Заполняет пиксели изображения на основе двух масок.
    Явно использует евкидово расстояние.

    Параметры:
        img: np.array формы (height, width, ch) - исходное изображение
        mask_1: np.array формы (height, width) - первая булева маска
        mask_2: np.array формы (height, width) - вторая булева маска

    Возвращает:
        Обработанное изображение той же формы
    """
    # Создаем копию изображения для модификации
    result = img.copy()
    height, width, ch = img.shape

    # Случай 1: mask_1=True и mask_2=False
    # Заполняем ближайшими пикселями где mask_1=False
    condition1 = mask_1 & ~mask_2
    if np.any(condition1):
        """
        for c in range(ch):
            result[condition1, c] = 0.0
        """
        # Создаем маску источников (где брать значения)
        sources1 = ~mask_1
        for c in range(ch):
            # Получаем индексы ближайших пикселей
            _, indices = distance_transform_edt(~sources1, return_indices=True)
            # Заполняем пиксели
            result[condition1, c] = img[
                indices[0][condition1], indices[1][condition1], c
            ]

    # Случай 2: mask_1=False и mask_2=True
    # Заполняем ближайшими пикселями где mask_1=True
    condition2 = mask_2  # ~mask_1 & mask_2
    if np.any(condition2):
        # Создаем маску источников (где брать значения)
        for c in range(ch):
            result[condition2, c] = 0.0
        """    
        sources2 = mask_1
        for c in range(ch):
            # Получаем индексы ближайших пикселей
            _, indices = distance_transform_edt(~sources2, return_indices=True)
            # Заполняем пиксели
            result[condition2, c] = img[
                indices[0][condition2], indices[1][condition2], c
            ]
         """
    return result


def create_coef(vars_m):
    structure = {
        "snwice_mass": {
            "f": "v_i",
        },
        "snwice_mass_b": {
            "f": "v_i",
        },
        "v_s": {
            "f": "v_i",
        },
        "sv_i": {
            "f": "v_i",
        },
        "e_s_l01": {
            "f": "v_i",
        },
        "e_i_l01": {
            "f": "v_i",
        },
        "e_i_l02": {
            "f": "v_i",
        },
        "stress1_i": {
            "f": "v_i",
        },
        "oa_i": {
            "f": "a_i",
        },
    }
    for key in structure:
        key_f = structure[key]["f"]

        x = vars_m[key_f].reshape(-1, 1)
        y = vars_m[key].reshape(-1, 1)

        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)

        structure[key]["coef"] = model.coef_[0][0]
    return structure


def select_keys(pieces):
    with xr.open_dataset(pieces[0]) as ds:
        keys = [
            var for var in ds.variables if ds[var].dims == ("time_counter", "y", "x")
        ]
        keys.extend(
            [
                var
                for var in ds.variables
                if ds[var].dims == ("time_counter", "numcat", "y", "x")
            ]
        )
    return keys


def write_files(pieces, vars_pieces_output, folder):
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir()
    for piece in pieces:
        with xr.open_dataset(piece) as ds:
            for key in vars_pieces_output.keys():
                name = f"index_{Path(piece).name.split('_')[-1][:-3]}"
                if len(ds[key].data.shape) == 3 and ds[key].data.shape[0] == 1:
                    ds[key].data[0] = vars_pieces_output[key][name]
                elif len(ds[key].data.shape) == 4 and ds[key].data.shape[0] == 1 and ds[key].data.shape[1] == 1:
                    ds[key].data[0][0] = vars_pieces_output[key][name]
                else:
                    ds[key].data = vars_pieces_output[key][name]
            out_file = folder / Path(piece).name
            if out_file.exists():
                os.remove(out_file)
            ds.to_netcdf(out_file)
    print("Restart has been rewrited! Success! ^_^")


def fill_nn_by_masks(
    img, mask_starting_values, mask_new_values, to_fill_with_zeros=False
):
    result = img.copy()
    height, width, ch = img.shape

    condition1 = mask_new_values & ~mask_starting_values
    if np.any(condition1):
        for c in range(ch):
            _, indices = distance_transform_edt(
                ~mask_starting_values, return_indices=True
            )
            result[condition1, c] = img[
                indices[0][condition1], indices[1][condition1], c
            ]

    condition2 = ~mask_new_values & mask_starting_values
    if np.any(condition2):
        for c in range(ch):
            _, indices = distance_transform_edt(
                mask_starting_values, return_indices=True
            )
            if to_fill_with_zeros:
                result[condition2, c] = 0.0
            else:
                result[condition2, c] = img[
                    indices[0][condition2], indices[1][condition2], c
                ]
    return result


def smooth_north_pole(variable, kernel, start_ind=(750, 780), end_ind=(910, 900)):

    start_ind = (750, 780)
    end_ind = (910, 900)
    region = variable[start_ind[0] : end_ind[0], start_ind[1] : end_ind[1]]
    region = convolve2d(region, kernel, mode="same", boundary="symm")
    variable[start_ind[0] : end_ind[0], start_ind[1] : end_ind[1]] = region
    return variable


def _piece_index_name(piece):
    return f"index_{Path(piece).name.split('_')[-1][:-3]}"


def _select_ice_restart_pieces(restart_dir):
    return [
        str(Path(restart_dir) / i)
        for i in os.listdir(restart_dir)
        if i.find("restart_ice") >= 0
    ]


def _select_ocean_restart_pieces(restart_dir):
    restart_dir = Path(restart_dir)
    pieces = []
    for name in os.listdir(restart_dir):
        if "restart_ice" in name or not name.endswith(".nc"):
            continue
        if name.startswith("restart-") or name.startswith("restart_") or name == "restart.nc":
            pieces.append(str(restart_dir / name))
    return sorted(pieces)


def _load_ice_restart_state(restart_dir, domain, keys=("a_i", "v_i")):
    pieces_ice_restart = _select_ice_restart_pieces(restart_dir)
    if len(pieces_ice_restart) == 0:
        raise FileNotFoundError(f"No restart_ice files found in {restart_dir}")

    mapping = create_mapping(pieces_ice_restart, domain)
    vars_ice = load_data(pieces_ice_restart, mapping, keys)
    return {key: np.asarray(value).squeeze() for key, value in vars_ice.items()}


def _match_grid_shape(field, shape, name):
    field = np.asarray(field, dtype=np.float32)
    if field.shape == shape:
        return field
    if field.ndim == 2 and field.T.shape == shape:
        return field.T
    raise ValueError(f"{name} shape {field.shape} does not match restart grid shape {shape}")


def _existing_restart_vars(pieces, candidates):
    if len(pieces) == 0:
        return []
    with xr.open_dataset(pieces[0]) as ds:
        return [key for key in candidates if key in ds.variables]


def _load_restart_vars_full(pieces, mapping, keys):
    data = {key: {} for key in keys}
    for piece in pieces:
        name = _piece_index_name(piece)
        with xr.open_dataset(piece) as ds:
            for key in keys:
                if key not in ds.variables:
                    continue
                array = np.asarray(ds[key].data)
                dims = ds[key].dims
                first_dim = str(dims[0]).lower() if len(dims) > 0 else ""
                if array.ndim >= 3 and array.shape[0] == 1 and "time" in first_dim:
                    array = array[0]
                elif array.ndim == 4 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim < 2:
                    continue
                data[key][name] = array

    output = {}
    for key, pieces_data in data.items():
        if len(pieces_data) > 0:
            output[key] = map_pieces(pieces_data, mapping)
    return output


def _write_restart_vars_full(pieces, vars_pieces_output, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    for piece in pieces:
        name = _piece_index_name(piece)
        with xr.open_dataset(piece) as ds_in:
            ds = ds_in.load()
        for key, pieces_output in vars_pieces_output.items():
            if key not in ds.variables or name not in pieces_output:
                continue
            output = pieces_output[name]
            target = ds[key].data
            if target.shape == output.shape:
                ds[key].data = output
            elif target.ndim == output.ndim + 1 and target.shape[0] == 1:
                ds[key].data[0] = output
            else:
                raise ValueError(
                    f"Cannot write {key}: restart shape {target.shape}, output shape {output.shape}"
                )

        out_file = folder / Path(piece).name
        if out_file.exists():
            os.remove(out_file)
        ds.to_netcdf(out_file)
        ds.close()


def _smooth_weight(weight, passes):
    weight = np.asarray(weight, dtype=np.float32)
    passes = int(passes)
    if passes <= 0:
        return np.clip(weight, 0.0, 1.0)
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    smoothed = weight
    for _ in range(passes):
        smoothed = convolve2d(smoothed, kernel, mode="same", boundary="symm")
    return np.clip(smoothed, 0.0, 1.0)


def _surface_field(field):
    field = np.asarray(field)
    if field.ndim == 2:
        return field
    if field.ndim == 3:
        return field[0]
    raise ValueError(f"Expected 2D or 3D restart variable, got shape {field.shape}")


def _vertical_taper(update_levels, max_levels, vertical_taper=None):
    levels = min(int(update_levels), int(max_levels))
    if levels <= 0:
        return np.asarray([], dtype=np.float32)
    if vertical_taper is None:
        return np.linspace(1.0, 0.25, levels, dtype=np.float32)

    taper = np.asarray(vertical_taper, dtype=np.float32).ravel()
    if taper.size == 0:
        return np.linspace(1.0, 0.25, levels, dtype=np.float32)
    if taper.size < levels:
        taper = np.pad(taper, (0, levels - taper.size), mode="edge")
    return taper[:levels]


def _apply_surface_increment(variable, increment, update_levels, vertical_taper=None):
    variable = variable.copy()
    increment = np.asarray(increment, dtype=np.float32)
    if variable.ndim == 2:
        variable += increment
        return variable

    taper = _vertical_taper(update_levels, variable.shape[0], vertical_taper)
    for level, coef in enumerate(taper):
        variable[level] += coef * increment
    return variable


def assimilate_ocean_surface_restarts(
    restart_dir,
    siconc,
    sosstsst,
    old_siconc,
    domain,
    ocean_config=None,
    ice_volume_increment=None,
):
    ocean_config = {} if ocean_config is None else ocean_config
    bounded = bool(ocean_config.get("bounded", False))
    # Keep volume-based selection and freezing safeguards unchanged when testing
    # the heat/salt caps. Switching to physical_ocean changes those rules too.
    apply_budget_limits = ocean_config.get("apply_budget_limits", True)
    if type(apply_budget_limits) is not bool:
        raise ValueError("apply_budget_limits must be a boolean")
    pieces_ocean_restart = _select_ocean_restart_pieces(restart_dir)
    strict = bool(ocean_config.get("strict", False))
    if len(pieces_ocean_restart) == 0:
        message = f"No ocean restart files found in {restart_dir}; skipping ocean update"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {"updated": False, "reason": "missing_ocean_restart"}

    temperature_candidates = tuple(ocean_config.get("temperature_vars", ("tn", "tb")))
    salinity_candidates = tuple(ocean_config.get("salinity_vars", ("sn", "sb")))
    temperature_vars = _existing_restart_vars(pieces_ocean_restart, temperature_candidates)
    salinity_vars = _existing_restart_vars(pieces_ocean_restart, salinity_candidates)
    if len(temperature_vars) == 0:
        message = (
            f"None of ocean temperature variables {temperature_candidates} found in "
            f"{pieces_ocean_restart[0]}; skipping ocean update"
        )
        if strict:
            raise KeyError(message)
        print(message)
        return {"updated": False, "reason": "missing_temperature_vars"}

    try:
        mapping = create_mapping(pieces_ocean_restart, domain)
        vars_ocean = _load_restart_vars_full(
            pieces_ocean_restart, mapping, temperature_vars + salinity_vars,
        )
        reference_temp = vars_ocean[temperature_vars[0]]
        surface_temp = _surface_field(reference_temp)
        grid_shape = surface_temp.shape

        old_a = _match_grid_shape(old_siconc, grid_shape, "old_siconc")
        new_a = _match_grid_shape(siconc, grid_shape, "siconc")
        analysis_sst = _match_grid_shape(sosstsst, grid_shape, "sosstsst")
        if bounded:
            if ice_volume_increment is None:
                raise ValueError("bounded ocean restart update requires ice_volume_increment")
            delta_v_i = _match_grid_shape(
                ice_volume_increment, grid_shape, "ice_volume_increment",
            )
        else:
            delta_v_i = None
    except Exception as exc:
        message = f"Could not prepare ocean restart update: {type(exc).__name__}: {exc}"
        if strict:
            raise
        print(f"{message}; skipping ocean update")
        return {"updated": False, "reason": "prepare_failed", "error": str(exc)}
    new_a = np.where(np.isnan(new_a) | (new_a < 0.01), 0.0, new_a)
    new_a = np.clip(new_a, 0.0, 1.0)
    old_a = np.where(np.isfinite(old_a), old_a, 0.0)
    if delta_v_i is not None:
        delta_v_i = np.where(np.isfinite(delta_v_i), delta_v_i, 0.0).astype(np.float32)

    delta_a = new_a - old_a
    delta_threshold = float(ocean_config.get("ice_delta_threshold", 0.05))
    changed_ice = np.abs(delta_a) >= delta_threshold
    if bounded:
        volume_threshold = float(ocean_config.get("ice_volume_delta_threshold", 0.001))
        changed_ice = changed_ice | (np.abs(delta_v_i) >= volume_threshold)
    finite_surface = np.isfinite(surface_temp)
    finite_sst = np.isfinite(analysis_sst)
    active = changed_ice & finite_surface
    if not np.any(active):
        print("No finite changed-ice cells found for ocean restart update; skipping ocean update")
        return {"updated": False, "reason": "no_active_cells"}

    weight = _smooth_weight(
        active.astype(np.float32),
        ocean_config.get("smooth_passes", 1),
    )
    weight *= finite_surface.astype(np.float32)

    default_freezing_temp = float(ocean_config.get("freezing_temp", -1.8))
    if len(salinity_vars) > 0 and salinity_vars[0] in vars_ocean:
        try:
            surface_salinity = _surface_field(vars_ocean[salinity_vars[0]])
            surface_salinity = _match_grid_shape(surface_salinity, grid_shape, salinity_vars[0])
            freezing_temp = -0.054 * np.where(np.isfinite(surface_salinity), surface_salinity, 34.0)
        except Exception as exc:
            if strict:
                raise
            print(
                f"Could not use salinity for freezing temperature ({type(exc).__name__}: {exc}); "
                "using configured default"
            )
            freezing_temp = np.full(grid_shape, default_freezing_temp, dtype=np.float32)
    else:
        freezing_temp = np.full(grid_shape, default_freezing_temp, dtype=np.float32)

    max_cool = float(ocean_config.get("max_cool", 0.25))
    max_warm = float(ocean_config.get("max_warm", 0.15))
    alpha = float(ocean_config.get("alpha", 0.25))
    ice_gain_buffer = float(ocean_config.get("ice_gain_buffer", 0.05))
    ice_loss_buffer = float(ocean_config.get("ice_loss_buffer", 0.10))

    target = surface_temp.copy()
    sst_increment = np.clip(analysis_sst - surface_temp, -max_cool, max_warm)
    target[finite_sst] = surface_temp[finite_sst] + sst_increment[finite_sst]

    ice_gain = active & (delta_a > 0.0)
    ice_loss = active & (delta_a < 0.0)
    if bounded:
        ice_gain = active & (delta_v_i > volume_threshold)
        ice_loss = active & (delta_v_i < -volume_threshold)
    target[ice_gain] = np.minimum(target[ice_gain], freezing_temp[ice_gain] + ice_gain_buffer)
    target[ice_loss] = np.maximum(target[ice_loss], freezing_temp[ice_loss] + ice_loss_buffer)

    increment = alpha * weight * np.clip(target - surface_temp, -max_cool, max_warm)
    increment[~finite_surface] = 0.0

    bounded_limits = {}
    if bounded:
        rho0 = float(ocean_config.get("reference_density", 1025.0))
        cp = float(ocean_config.get("heat_capacity", 4000.0))
        rho_ice = float(ocean_config.get("ice_density", 900.0))
        latent_heat = float(ocean_config.get("latent_heat_fusion", 334000.0))
        effective_depth = max(float(ocean_config.get("effective_depth_m", 20.0)), 1.0e-6)
        heat_safety = float(ocean_config.get("heat_safety_factor", 1.0))
        heat_safety_loss = float(
            ocean_config.get("heat_safety_factor_loss", heat_safety)
        )
        min_cooling = float(ocean_config.get("min_cooling_j_m2", 0.0))
        min_warming = float(ocean_config.get("min_warming_j_m2", min_cooling))

        cooling_limit = (
            heat_safety * rho_ice * latent_heat * np.maximum(delta_v_i, 0.0)
            + min_cooling
        ) / (rho0 * cp * effective_depth)
        warming_limit = (
            heat_safety_loss * rho_ice * latent_heat * np.maximum(-delta_v_i, 0.0)
            + min_warming
        ) / (rho0 * cp * effective_depth)
        if apply_budget_limits:
            increment = np.minimum(np.maximum(increment, -cooling_limit), warming_limit)

        supercool_tolerance = float(ocean_config.get("supercool_tolerance", 0.05))
        freezing_lower_increment = freezing_temp - supercool_tolerance - surface_temp
        increment[active] = np.maximum(
            increment[active], freezing_lower_increment[active]
        )
        increment[~finite_surface] = 0.0
        bounded_limits.update({
            "max_cooling_limit": float(np.nanmax(cooling_limit[active])),
            "max_warming_limit": float(np.nanmax(warming_limit[active])),
        })

    update_levels = int(ocean_config.get("update_levels", 1))
    vertical_taper = ocean_config.get("vertical_taper")
    temp_min = float(ocean_config.get("temperature_min", -3.0))
    temp_max = float(ocean_config.get("temperature_max", 35.0))

    vars_ocean_assim = {}
    for key in temperature_vars:
        updated = _apply_surface_increment(
            vars_ocean[key], increment, update_levels, vertical_taper=vertical_taper,
        )
        vars_ocean_assim[key] = np.clip(updated, temp_min, temp_max)

    salinity_update = bool(ocean_config.get("salinity_update", False))
    salinity_increment = None
    if salinity_update:
        if len(salinity_vars) == 0:
            message = "salinity_update=true but no configured salinity variables were found"
            if strict:
                raise KeyError(message)
            print(f"{message}; skipping salinity update")
        else:
            salinity_alpha = float(ocean_config.get("salinity_alpha", 1.0))
            max_salt_increase = float(ocean_config.get("max_salt_increase", 0.05))
            max_salt_decrease = float(ocean_config.get("max_salt_decrease", 0.05))
            salinity_min = float(ocean_config.get("salinity_min", 0.0))
            salinity_max = float(ocean_config.get("salinity_max", 40.0))

            salinity_target_increment = np.zeros(grid_shape, dtype=np.float32)
            salinity_target_increment[ice_gain] = max_salt_increase
            salinity_target_increment[ice_loss] = -max_salt_decrease
            salinity_increment = salinity_alpha * weight * salinity_target_increment
            if bounded:
                effective_depth = max(
                    float(ocean_config.get("effective_depth_m", 20.0)), 1.0e-6,
                )
                rho0 = float(ocean_config.get("reference_density", 1025.0))
                rho_ice = float(ocean_config.get("ice_density", 900.0))
                salt_safety = float(ocean_config.get("salt_safety_factor", 1.0))
                salt_safety_loss = float(
                    ocean_config.get("salt_safety_factor_loss", salt_safety)
                )
                seawater_salinity = float(ocean_config.get("seawater_salinity_ref", 34.0))
                ice_salinity = float(ocean_config.get("ice_salinity_ref", 7.0))
                salinity_contrast = max(seawater_salinity - ice_salinity, 0.0)
                min_salt_increase = float(ocean_config.get("salt_min_increase_psu", 0.0))
                min_salt_decrease = float(
                    ocean_config.get("salt_min_decrease_psu", min_salt_increase)
                )

                salt_increase_limit = (
                    min_salt_increase
                    + salt_safety
                    * rho_ice
                    * np.maximum(delta_v_i, 0.0)
                    * salinity_contrast
                    / (rho0 * effective_depth)
                )
                salt_decrease_limit = (
                    min_salt_decrease
                    + salt_safety_loss
                    * rho_ice
                    * np.maximum(-delta_v_i, 0.0)
                    * salinity_contrast
                    / (rho0 * effective_depth)
                )
                if apply_budget_limits:
                    salinity_increment = np.minimum(
                        np.maximum(salinity_increment, -salt_decrease_limit),
                        salt_increase_limit,
                    )
                bounded_limits.update({
                    "max_salt_increase_limit": float(np.nanmax(salt_increase_limit[active])),
                    "max_salt_decrease_limit": float(np.nanmax(salt_decrease_limit[active])),
                })
            salinity_increment[~finite_surface] = 0.0

            for key in salinity_vars:
                updated = _apply_surface_increment(
                    vars_ocean[key], salinity_increment, update_levels,
                    vertical_taper=vertical_taper,
                )
                vars_ocean_assim[key] = np.clip(updated, salinity_min, salinity_max)

    vars_pieces_output = {
        key: map_pieces(value, mapping)
        for key, value in vars_ocean_assim.items()
    }
    _write_restart_vars_full(pieces_ocean_restart, vars_pieces_output, restart_dir)
    bound_message = f", bounded_limits={bounded_limits}" if bounded_limits else ""
    print(
        "Ocean restart surface update applied: "
        f"vars={temperature_vars}, active_cells={int(active.sum())}, "
        f"max_abs_temp_increment={float(np.nanmax(np.abs(increment))):.6g}, "
        f"bounded={bounded}, "
        f"budget_limits_applied={bounded and apply_budget_limits}, "
        f"salinity_update={salinity_update}"
        f"{bound_message}"
    )
    max_abs_salt_increment = None
    if salinity_increment is not None:
        max_abs_salt_increment = float(np.nanmax(np.abs(salinity_increment)))
    return {
        "updated": True,
        "temperature_vars": temperature_vars,
        "salinity_vars": salinity_vars,
        "active_cells": int(active.sum()),
        "max_abs_increment": float(np.nanmax(np.abs(increment))),
        "salinity_update": salinity_update,
        "max_abs_salt_increment": max_abs_salt_increment,
        "bounded": bounded,
        "budget_limits_applied": bounded and apply_budget_limits,
        "bounded_limits": bounded_limits,
    }


def assimilate_restarts_physical_ocean(
    restart_dir,
    siconc,
    sithic,
    sosstsst,
    domain,
    ocean_config=None,
):
    old_ice = _load_ice_restart_state(restart_dir, domain, keys=("a_i",))
    assimilate_restarts_physical(restart_dir, siconc, sithic, domain)
    return assimilate_ocean_surface_restarts(
        restart_dir,
        siconc,
        sosstsst,
        old_ice["a_i"],
        domain,
        ocean_config=ocean_config,
    )


def assimilate_restarts_physical_ocean_bounded(
    restart_dir,
    siconc,
    sithic,
    sosstsst,
    domain,
    ocean_config=None,
):
    old_ice = _load_ice_restart_state(restart_dir, domain, keys=("a_i", "v_i"))
    ice_scheme = (ocean_config or {}).get("ice_energy_scheme", "legacy")
    ice_temperature_k = (ocean_config or {}).get("new_ice_temperature_k", 270.0)
    if ice_scheme == "legacy":
        assimilate_restarts_physical(restart_dir, siconc, sithic, domain)
    else:
        assimilate_restarts_physical(
            restart_dir, siconc, sithic, domain,
            ice_energy_scheme=ice_scheme, new_ice_temperature_k=ice_temperature_k,
        )
    new_ice = _load_ice_restart_state(restart_dir, domain, keys=("v_i",))

    bounded_config = {} if ocean_config is None else dict(ocean_config)
    bounded_config["bounded"] = True
    return assimilate_ocean_surface_restarts(
        restart_dir,
        siconc,
        sosstsst,
        old_ice["a_i"],
        domain,
        ocean_config=bounded_config,
        ice_volume_increment=new_ice["v_i"] - old_ice["v_i"],
    )



def assimilate_restarts_naively(
    restart_dir,
    siconc,
    sithic,
    domain,
    keys_assim=("a_i", "v_i"),
    key_mask="a_i",
):

    pieces_ice_restart = [
        str(Path(restart_dir) / i)
        for i in os.listdir(restart_dir)
        if i.find("restart_ice") >= 0
    ]

    mapping = create_mapping(pieces_ice_restart, domain)
    keys_nemo = select_keys(pieces_ice_restart)
    keys_nemo.remove("t_su")
    vars_nemo_ice = load_data(pieces_ice_restart, mapping, keys_nemo)

    nemo_mask = vars_nemo_ice['a_i']==0
    new_mask = np.logical_or(
        np.isnan(siconc),
        siconc<0.01
    )
    siconc[new_mask] = 0.0
    siconc = np.clip(siconc, 0, 1)

    vars_nemo_assim = {}
    # Концентрация постобработка
    vars_nemo_assim['a_i'] = siconc.copy()
    
    # Удельная толщина льда
    mask = np.logical_or(
        np.isnan(sithic),
        sithic<0.00
    )
    vars_nemo_assim['v_i'] = sithic.copy()
    vars_nemo_assim['v_i'][mask] = 0.0
    vars_nemo_assim['v_i'][new_mask] = 0.0
    # если концентрация не нулевая - ставим минимум 10 см льда
    mask = np.logical_and(
        vars_nemo_assim['v_i']==0,
        vars_nemo_assim['a_i']!=0
    )
    vars_nemo_assim['v_i'][mask] = 0.1
    
    vars_nemo_assim['v_i'] = (
        vars_nemo_assim['v_i'].copy()*
        vars_nemo_assim['a_i'].copy()
    )

    arrays = []
    shapes_from = []
    for key in keys_nemo:
        shapes_from.append(vars_nemo_ice[key].shape)
        vars_nemo_ice[key] = vars_nemo_ice[key].squeeze()

    mask_from = vars_nemo_ice[key_mask] > 0
    mask_assim = siconc > 0.0

    for key in keys_nemo:
        array = vars_nemo_ice[key]
        arrays.append(array)
    nemo_array = np.stack(arrays, axis=-1)

    corr_f = create_coef(vars_nemo_ice)

    an_array = ed_nn_1(nemo_array, mask_from, mask_assim)
    vars_an = {key: an_array[:, :, i] for i, key in enumerate(keys_nemo)}
    for key in keys_assim:
        vars_an[key] = vars_nemo_assim[key]
    for key in corr_f:
        f = corr_f[key]
        vars_an[key] = vars_nemo_assim[f["f"]] * f["coef"]

    vars_output = {}
    vars_pieces_output = {}
    for i, key in enumerate(keys_nemo):
        vars_output[key] = np.reshape(vars_an[key], shapes_from[i])

    for key in vars_output.keys():
        vars_pieces_output[key] = map_pieces(vars_output[key], mapping)
    
    write_files(pieces_ice_restart, vars_pieces_output, restart_dir)


def assimilate_restarts_physical(
    restart_dir,
    siconc,
    sithic,
    domain,
    ice_energy_scheme="legacy",
    new_ice_temperature_k=270.0,
):

    if ice_energy_scheme not in ("legacy", "si3_native"):
        raise ValueError(f"Unknown ice energy scheme: {ice_energy_scheme}")

    with xr.open_dataset(domain) as ds:
        mask_land = ds.top_level.data[0] == 0

    pieces_ice_restart = [
        str(Path(restart_dir) / i)
        for i in os.listdir(restart_dir)
        if i.find("restart_ice") >= 0
    ]
    
    mapping = create_mapping(pieces_ice_restart, domain)
    keys_nemo = select_keys(pieces_ice_restart)
    keys_nemo.remove("t_su")
    vars_nemo_ice = load_data(pieces_ice_restart, mapping, keys_nemo)

    nemo_mask = vars_nemo_ice['a_i']==0
    new_mask = np.logical_or(
        np.isnan(siconc),
        siconc<0.01
    )

    vars_nemo_assim = {}
    # Концентрация постобработка
    vars_nemo_assim['a_i'] = siconc.copy()
    vars_nemo_assim['a_i'][new_mask] = 0.0
    vars_nemo_assim['a_i'] = np.clip(vars_nemo_assim['a_i'], 0, 1)
    
    # Удельная толщина льда
    mask = np.logical_or(
        np.isnan(sithic),
        sithic<0.00
    )
    vars_nemo_assim['v_i'] = sithic.copy()
    vars_nemo_assim['v_i'][mask] = 0.0
    vars_nemo_assim['v_i'][new_mask] = 0.0
    # если концентрация не нулевая - ставим минимум 10 см льда
    mask = np.logical_and(
        vars_nemo_assim['v_i']==0,
        vars_nemo_assim['a_i']!=0
    )
    vars_nemo_assim['v_i'][mask] = 0.1
    
    vars_nemo_assim['v_i'] = (
        vars_nemo_assim['v_i'].copy()*
        vars_nemo_assim['a_i'].copy()
    )

    # Убираем снег, там где лед растаял
    snthic = vars_nemo_ice['v_s'].copy()
    mask = vars_nemo_ice['a_i']!=0
    # снег не может лежать на воде
    snthic[mask] = (
        vars_nemo_ice['v_s'][mask].copy() /
        vars_nemo_ice['a_i'][mask].copy()
    )
    snthic[new_mask] = 0.0
    vars_nemo_assim['v_s'] = snthic*vars_nemo_assim['a_i'].copy()

    # Честно считаем массу льда и снега
    rho_ice = 917 # kg/m3 (плотность льда)
    rho_snow = 330 # kg/m3 (плотность снега)
    vars_nemo_assim['snwice_mass'] = (
        rho_ice*vars_nemo_assim['v_i'].copy() +
        rho_snow*vars_nemo_assim['v_s'].copy()
    )
    vars_nemo_assim['snwice_mass_b'] = (
        rho_ice*vars_nemo_assim['v_i'].copy() +
        rho_snow*vars_nemo_assim['v_s'].copy()
    )
    
    # Разбираемся с соленостью
    s_ref = 7.4 # адекватное значение солености для нового льда
    vars_nemo_assim['sv_i'] = vars_nemo_ice['sv_i'].copy()
    mask = vars_nemo_ice['v_i']!=0 # перенормируем на новое колличество льда
    vars_nemo_assim['sv_i'][mask] = (
        vars_nemo_assim['sv_i'][mask].copy()*
        vars_nemo_assim['v_i'][mask].copy()/
        vars_nemo_ice['v_i'][mask].copy()
    )
    vars_nemo_assim['sv_i'][new_mask] = 0.0
    mask = np.logical_and(
        vars_nemo_assim['sv_i']==0,
        vars_nemo_assim['a_i']!=0
    )
    vars_nemo_assim['sv_i'][mask] = (
        vars_nemo_assim['v_i'][mask] *
        s_ref *
        rho_ice/1_000
    )
    
    # Разбираемся с энергиями
    d_t = 2 # делаем зазор по температуре, чтобы лед не таял сразу
    c_i = 2100 # J/kg·K (удельная теплоемкость льда)
    l_i = 33_400 # J/kg (удельная теплота плавления льда)
    c_s = 2090 # J/kg·K (удельная теплоемкость снега)
    vars_nemo_assim['e_s_l01'] = vars_nemo_ice['e_s_l01'].copy()
    mask = vars_nemo_ice['v_s']!=0 # перенормируем на новое колличество снега
    vars_nemo_assim['e_s_l01'][mask] = (
        vars_nemo_assim['e_s_l01'][mask].copy()*
        vars_nemo_assim['v_s'][mask].copy()/
        vars_nemo_ice['v_s'][mask].copy()
    )
    vars_nemo_assim['e_s_l01'][new_mask] = 0.0 # Разбираемся с энергиями
    mask = np.logical_and(
        vars_nemo_ice['v_s']==0,
        vars_nemo_assim['v_s']!=0
    ) # новый снег (вроде у нас такого нет, но вдруг)
    vars_nemo_assim['e_s_l01'][mask] = (
        vars_nemo_assim['v_s'][mask].copy()*rho_snow*l_i +
        vars_nemo_assim['v_s'][mask].copy()*rho_snow*c_s*d_t
    )
    vars_nemo_assim['e_i_l01'] = vars_nemo_ice['e_i_l01'].copy()
    vars_nemo_assim['e_i_l02'] = vars_nemo_ice['e_i_l02'].copy()

    mask = vars_nemo_ice['v_i']!=0 # перенормируем на новое колличество льда
    vars_nemo_assim['e_i_l01'][mask] = (
        vars_nemo_assim['e_i_l01'][mask].copy()*
        vars_nemo_assim['v_i'][mask].copy()/
        vars_nemo_ice['v_i'][mask].copy()
    )
    vars_nemo_assim['e_i_l02'][mask] = (
        vars_nemo_assim['e_i_l02'][mask].copy()*
        vars_nemo_assim['v_i'][mask].copy()/
        vars_nemo_ice['v_i'][mask].copy()
    )
    vars_nemo_assim['e_i_l01'][new_mask] = 0.0
    vars_nemo_assim['e_i_l02'][new_mask] = 0.0

    mask = np.logical_and(
        vars_nemo_ice['v_i']==0,
        vars_nemo_assim['v_i']!=0
    ) # новый снег (вроде у нас такого нет, но вдруг)
    if ice_energy_scheme == "si3_native":
        # NEMO iceistate.F90:373-376, nn_icesal=2. The updater stores
        # sv_i/v_i = s_ref*rho_ice/1000; this exceeds 4.5 g/kg, so the
        # SI3 vertical salinity profile is constant. Two equal layers.
        cold_content = new_ice_energy_density(
            new_ice_temperature_k, s_ref * rho_ice / 1000.0)
        for key in ('e_i_l01', 'e_i_l02'):
            vars_nemo_assim[key][mask] = vars_nemo_assim['v_i'][mask] * cold_content / 2.0
    else:
        vars_nemo_assim['e_i_l01'][mask] = (
            vars_nemo_assim['v_i'][mask].copy()*rho_ice*c_i +
            vars_nemo_assim['v_i'][mask].copy()*rho_ice*c_s*d_t
        )/2
        vars_nemo_assim['e_i_l02'][mask] = (
            vars_nemo_assim['v_i'][mask].copy()*rho_ice*c_i +
            vars_nemo_assim['v_i'][mask].copy()*rho_ice*c_s*d_t
        )/2
    
    # Применяем ближайшего соседа
    nemo_ice_array = np.stack([
        vars_nemo_ice[key] for key in keys_nemo
    ], axis=-1)
    nn_ice_array = ed_nn_1(nemo_ice_array, nemo_mask, new_mask)
    vars_nemo_nn = {
        key: nn_ice_array[:, :, i]
        for i, key in enumerate(keys_nemo)
    }
    for key in vars_nemo_nn:
        if key not in vars_nemo_assim:
            vars_nemo_assim[key] = vars_nemo_nn[key].copy()
    
    vars_pieces_output = {}
    for key in vars_nemo_assim:
        vars_pieces_output[key] = map_pieces(vars_nemo_assim[key], mapping)

    write_files(pieces_ice_restart, vars_pieces_output, restart_dir)
