import numpy as np

class Activation:
    def __init__(self, func: callable, derivative: callable):
        self.func = func
        self.derivative = derivative
    
    def __call__(self, x):
        return self.func(x)

relu = Activation(lambda x: np.maximum(0, x), lambda x: x > 0)
sigmoid = Activation(lambda x: 1 / (1 + np.exp(-x)), lambda x: 1 / (1 + np.exp(-x)) * 1 / (1 + np.exp(x-1)))
