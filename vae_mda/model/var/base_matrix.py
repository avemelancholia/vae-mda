import torch
import numpy as np

from abc import ABC, abstractmethod


class CorrelationMatrix(ABC):
    name = ''
    def __init__(self, path='', epsilon=0.002):
        super().__init__()
        self.epsilon = epsilon
        self.path = path
        self.matrix = torch.tensor([]).cpu()
        self.inv_matrix = torch.tensor([])

    @abstractmethod
    def calculate_matrix(*args, **kwargs):
        pass

    def calculate_inv_matrix(self):
        try:
            # check positive define
            torch.linalg.cholesky(
                (self.matrix + self.epsilon * torch.eye(self.matrix.shape[0]).cpu())
            )
        except torch._C._LinAlgError:
            raise Exception("Matrix is not positive definite. "
                            "Regulation is not enouth: increase epsilon")
        
        print("Invert correlation matrix")
        print(self.matrix.shape)
        self.inv_matrix = torch.linalg.inv(
            (self.matrix.cpu() + self.epsilon * torch.eye(self.matrix.shape[0]).cpu())
        ).to(torch.tensor([0]).device)
        np.save(self.path, self.inv_matrix.cpu().numpy())
