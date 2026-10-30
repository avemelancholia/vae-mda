import numpy as np


def transform_1(arr):
    arr = arr[1]
    arr = np.round(arr, decimals=0)
    arr[arr==2.] = 1.
    arr[arr==3.] = np.nan
    return arr

def mask_transform(arr):
    arr = arr[1]
    arr = np.round(arr, decimals=0)
    arr[arr==2.] = 1.
    arr[arr==3.] = 1.
    return arr

def transform_8(arr):
    amsr = arr[7].copy()
    ice_cl = mask_transform(arr)
    mask = np.isnan(ice_cl)
    amsr[mask] = np.nan
    return amsr

def transform_9(arr):
    amsr = arr[7].copy()
    ice_cl = mask_transform(arr) 
    return amsr*ice_cl

def transform_10(arr):
    ice_cl = arr[1]
    ice_cl = np.round(ice_cl, decimals=0)
    amsr = arr[7].copy()
    mask = np.isnan(ice_cl)
    amsr[mask] = np.nan
    mask = (ice_cl==1.)
    amsr[mask] = 1.
    mask = (ice_cl==2.)
    amsr[mask] = 1.
    return amsr

def transform_selected(arr):
    ice_cl = arr[1]
    ice_cl = np.round(ice_cl, decimals=0)
    amsr = arr[7].copy()
    mask = np.isnan(ice_cl)
    amsr[mask] = np.nan
    mask = (ice_cl==0.)
    amsr[mask] = 0.
    
    condition_mask = (amsr == 0) & ((ice_cl == 1) | (ice_cl == 2))
    nonzero_mask = ~np.isnan(amsr) & (amsr != 0)
    
    if not np.any(condition_mask) or not np.any(nonzero_mask):
        return amsr
    
    # Получаем индексы элементов, которые нужно заменить
    replace_indices = np.where(condition_mask)
    
    # Получаем индексы ненулевых элементов
    nonzero_indices = np.where(nonzero_mask)
    nonzero_values = amsr[nonzero_mask]
    
    # Для каждого заменяемого элемента находим ближайший ненулевой элемент
    for idx in zip(*replace_indices):
        # Вычисляем расстояния до всех ненулевых элементов
        distances = np.sqrt(np.sum((np.array(nonzero_indices).T - idx)**2, axis=1))
        
        # Находим индекс ближайшего ненулевого элемента
        nearest_idx = np.argmin(distances)
        
        # Заменяем значение
        amsr[idx] = nonzero_values[nearest_idx]

    return amsr

def transform_12(arr):
    return arr

def transform_13(arr):
    amsr_track = transform_8(arr)
    hard_corrected = transform_10(arr)
    return 0.5 * amsr_track + 0.5 * hard_corrected

def transform_14(arr):
    ice_cl = arr[1]
    ice_cl = np.round(ice_cl, decimals=0)
    amsr = arr[7].copy()
    out = np.zeros(amsr.shape)
    out.fill(np.nan)

    ice_threshold = 0.15
    finite_amsr = ~np.isnan(amsr)
    open_agree = (ice_cl == 0.) & finite_amsr & (amsr <= ice_threshold)
    ice_agree = ((ice_cl == 1.) | (ice_cl == 2.)) & finite_amsr & (amsr >= ice_threshold)
    out[open_agree | ice_agree] = amsr[open_agree | ice_agree]
    return out

def transform_15(arr):
    ice_cl = arr[1]
    ice_cl = np.round(ice_cl, decimals=0)
    amsr = arr[7].copy()
    out = np.zeros(amsr.shape)
    out.fill(np.nan)

    finite_amsr = ~np.isnan(amsr)
    open_water = (ice_cl == 0.) & finite_amsr
    ice = ((ice_cl == 1.) | (ice_cl == 2.)) & finite_amsr
    out[open_water] = 0.5 * amsr[open_water]
    out[ice] = 0.5 * amsr[ice] + 0.5
    return out

def confidence_15(arr):
    ice_cl = arr[1]
    ice_cl = np.round(ice_cl, decimals=0)
    amsr = arr[7].copy()
    out = np.zeros(amsr.shape)
    out.fill(np.nan)

    ice_threshold = 0.15
    finite_amsr = ~np.isnan(amsr)
    open_water = (ice_cl == 0.) & finite_amsr
    ice = ((ice_cl == 1.) | (ice_cl == 2.)) & finite_amsr
    out[open_water] = 0.5
    out[ice] = 0.5
    out[open_water & (amsr <= ice_threshold)] = 1.0
    out[ice & (amsr >= ice_threshold)] = 1.0
    return out
    
sral_disc = {
    0: {
        'name': 'sea_ice_concentration_20_ku',
    },
    1: {
        'name': 'surf_type_class_20_ku',
        'transform': transform_1,
    },
    2: {
        'name': 'surf_type_20_ku',
    },
    3: {
        'name': 'freeboard_20_ku',
    },
    4: {
        'name': 'sea_ice_ssha_20_ku',
    },
    5: {
        'name': 'int_sea_ice_ssha_20_ku',
    },
    6: {
        'name': 'surf_type_class_modified',
    },
    7: {
        'name': 'amsr2',
    },
    8: {
        'name': 'amsr2_masked',
        'transform': transform_8,
    },
    9: {
        'name': 'amsr2_masked_class_0',
        'transform': transform_9,
    },
    10: {
        'name': 'amsr2_masked_class_0_1_2',
        'transform': transform_10,
    },
    
    11: {
        'name': 'amsr2_adjust_class_0_1_2',
        'transform': transform_selected,
    },
    12: {
        'name': 'twins_2023_to_2022',
        'transform': transform_12,
    },
    13: {
        'name': 'amsr2_soft50_adjust_class_0_1_2',
        'transform': transform_13,
    },
    14: {
        'name': 'amsr2_sral_agreement_filter',
        'transform': transform_14,
    },
    15: {
        'name': 'amsr2_soft_class50_confidence',
        'transform': transform_15,
        'confidence': confidence_15,
    }
}
