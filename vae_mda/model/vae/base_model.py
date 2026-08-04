from abc import ABC, abstractmethod


class VAEModel(ABC):
    
    @abstractmethod
    def encode(self, x):
        pass
    
    @abstractmethod
    def decode(self, lat):
        pass

    @abstractmethod
    def from_pretrained(self, log_dir):
        pass