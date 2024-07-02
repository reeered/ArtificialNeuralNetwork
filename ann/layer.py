import numpy as np
from ann.activation import Activation, sigmoid, relu

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        if activation == sigmoid:
            # 使用Xavier Normal方法初始化权重和阈值
            std = np.sqrt(2. / (input_size + output_size))
            self.weights = np.random.normal(loc=0., scale=std, size=[input_size, output_size])
            self.bias = np.random.normal(loc=0., scale=std, size=[output_size])
        elif activation == relu:
            # 使用He Normal方法初始化权重和阈值
            std = np.sqrt(2. / input_size)
            self.weights = np.random.normal(loc=0., scale=std, size=[input_size, output_size])
            self.bias = np.random.normal(loc=0., scale=std, size=[output_size])
    
    def forward(self, x: np.ndarray):
        self.input = x
        self.h = np.dot(x, self.weights) + self.bias
        H = self.activation(self.h)
        return H