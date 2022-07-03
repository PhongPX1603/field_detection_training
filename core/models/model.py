from torch import nn
from abc import ABC, abstractmethod


class Module(ABC):
    def attach(self, frame, module_name):
        self.frame = frame
        self.frame[module_name] = self
        self.module_name = module_name

    @abstractmethod
    def init(self):
        pass
    

class Model(nn.Module, Module):
    def __init__(self):
        super(Model, self).__init__()

    def init(self):
        pass
