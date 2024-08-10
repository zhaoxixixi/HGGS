from abc import ABCMeta, abstractmethod

class Normal_Parameters(metaclass=ABCMeta):
    ode_model_name = 'Normal'
    @abstractmethod
    def assign(self):
        pass

    @abstractmethod
    def build_csv(self):
        pass

    @abstractmethod
    def _multiple(self):
        pass
    
class Normal_Equation(metaclass=ABCMeta):
    @abstractmethod
    def instantiate(cls, y, t, params: Normal_Parameters):
        pass
    