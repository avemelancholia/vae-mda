from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    
    @abstractmethod
    def transform(self, x):
        pass
    
    @abstractmethod
    def detransform(self, x):
        pass

    @abstractmethod
    def val_transform(self, x):
        pass

    @abstractmethod
    def val_sat_transform(self, y):
        pass
    
    @abstractmethod
    def obs_transform(self, y):
        pass 
    
    @abstractmethod
    def obs_model_transform(self, x, mask):
        pass
    
    @abstractmethod
    def obs_detransform(self, y):
        pass
