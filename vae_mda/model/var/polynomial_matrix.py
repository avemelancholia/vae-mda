import torch
import numpy as np

from tqdm import tqdm

from .base_matrix import CorrelationMatrix

R = 6378

def fill_mask(field, mask, filler=np.nan):
    field_mask = np.isnan(field)
    field[np.logical_and(~mask, field_mask)] = filler
    field[mask] = np.nan
    return field

def haversine_distance(
    lat_1: np.array, lon_1: np.array, lat_2: np.array, lon_2: np.array
):
    """
    Function to calculate distance:
    formula Haversine (https://en.m.wikipedia.org/wiki/Haversine_formula)
    """
    lat_1 = np.tile(np.expand_dims(lat_1, axis=1), (1, len(lat_2)))
    lon_1 = np.tile(np.expand_dims(lon_1, axis=1), (1, len(lon_2)))
    lat_1 = np.radians(lat_1)
    lon_1 = np.radians(lon_1)
    lat_2 = np.radians(lat_2)
    lon_2 = np.radians(lon_2)

    return (
        2
        * R
        * np.arcsin(
            np.sqrt(
                np.sin((lat_2 - lat_1) / 2) ** 2
                + np.cos(lat_1) * np.cos(lat_2) *
                np.sin((lon_2 - lon_1) / 2) ** 2
            )
        )
    )

class PolynomialMatrix(CorrelationMatrix):
    name = "PolynomialMatrix"
    def __init__(self, config, epsilon=0.01, calculate_inv=True):
        super().__init__(path=config["inv_corr_matrix_path"], epsilon=epsilon)
        self.limit = config["limit"]
        self.get_coordinate(config)
        self.calculate_matrix()
        if calculate_inv:
            self.calculate_inv_matrix()
    
    def get_coordinate(self, config):
        lat = np.load(config["lat_path"])
        lon = np.load(config["lon_path"])
        mask = np.load(config["mask_path"])
        lat, lon = fill_mask(lat, mask), fill_mask(lon, mask)
        self.lat, self.lon = lat[~mask].flatten(), lon[~mask].flatten()
    
    def calculate_matrix(self):
        print("Calculate Polynomial Correlation Matrix")
        self.matrix = torch.tensor([]).cpu()
        for i in tqdm(range(0, self.lat.shape[0], 200)):
            distances = haversine_distance(
                self.lat, self.lon,
                self.lat[i : i + 200], self.lon[i : i + 200],
            )
            self.matrix = (
                torch.hstack([self.matrix, torch.tensor(
                    self.polinomial_function(distances, self.limit),
                    dtype=torch.float,
                ).cpu(),]).cpu() if self.matrix.size
                else torch.tensor(
                    self.polinomial_function(distances, self.limit),
                    dtype=torch.float,
                ).cpu()
            )

    def polinomial_function(self, distances, limit):
        """
        Formula 7 from article:
        'Ensemble optimal interpolation schemes for assimilating Argo profiles
        into a hybrid coordinate ocean model'
        """
        ans = np.zeros_like(distances)
        # first condition
        mask = (distances <= limit)
        np.putmask(
            ans,
            mask,
            (
                -((distances / limit) ** 5) / 4
                + ((distances / limit) ** 4) / 2
                + 5 * ((distances / limit) ** 3) / 8
                - 5 * ((distances / limit) ** 2) / 3
                + 1
            ),
        )
        # second condition
        mask = (distances > limit) * (distances <= 2 * limit)
        np.putmask(
            ans,
            mask,
            (
                ((distances / limit) ** 5) / 12
                - ((distances / limit) ** 4) / 2
                + 5 * ((distances / limit) ** 3) / 8
                + 5 * ((distances / limit) ** 2) / 3
                - 5 * (distances / limit)
                + 4
                - 2 * (((distances + 0.000000001) / limit) ** (-1)) / 3
            ),
        )
        return ans