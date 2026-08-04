from abc import ABC, abstractmethod, abstractstaticmethod
from torch import nn 


class AssimilationModel(nn.Module, ABC):
    name = ''
    
    @abstractstaticmethod
    def preprocess_tensors(tensors, config):
        # transform before fit to model
        pass

    @abstractmethod
    def fit(tensors, steps, meta):
        pass

    @abstractstaticmethod
    def init_params(config):
        pass

    @abstractstaticmethod
    def get_x():
        pass

    @abstractstaticmethod
    def get_obs():
        pass
