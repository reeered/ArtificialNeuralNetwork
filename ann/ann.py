import numpy as np

from ann.layer import Layer

class ANN:
    def __init__(self, layers: list, lr: float = 0.001):
        for layer in layers:
            assert isinstance(layer, Layer)
        self.layers = layers
        self.lr = lr
    
    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def clear_grad(self):
        for layer in self.layers:
            layer.dW = np.zeros((layer.input_size, layer.output_size))
            layer.db = np.zeros(layer.output_size)

    def backpropagation(self, y_pred: np.ndarray, y_true: np.ndarray):
        sigma = [np.zeros(layer.output_size) for layer in self.layers]
        # 更新输出层权重
        for i in range(self.layers[-1].output_size):
            # 输出层单元i的输入
            h_i = self.layers[-1].h[i]
            # 对激活函数求导
            derivative_phy_h_i = self.layers[-1].activation.derivative(h_i)
            err = y_true[i] - y_pred[i]
            sigma[-1][i] = derivative_phy_h_i * err
            # 调整阈值更新量
            self.layers[-1].db[i] += self.lr * sigma[-1][i]
            for j in range(self.layers[-1].input_size):
                eta = self.lr
                H_j = self.layers[-1].input[j]
                # 调整权重更新量
                self.layers[-1].dW[j][i] += eta * sigma[-1][i] * H_j

        # 反向传播，逆序遍历隐藏层
        for idx, layer in enumerate(self.layers[-2::-1]):
            layer_idx = len(self.layers) - idx - 2 # 隐藏层索引
            # 更新隐藏层权重
            for j in range(layer.output_size):
                # 隐单元j的输入
                h_j= layer.h[j]
                # 对激活函数求导
                derivative_phy_h_j = layer.activation.derivative(h_j)
                sigma[layer_idx][j] = derivative_phy_h_j * np.sum([sigma[layer_idx+1][i] * self.layers[layer_idx+1].weights[j]])
                # 调整阈值更新量
                layer.db[j] += eta * sigma[layer_idx][j]
                for k in range(layer.input_size):
                    eta = self.lr
                    I_k = layer.input[k]
                    # 调整权重更新量
                    layer.dW[k][j] += eta * sigma[layer_idx][j] * I_k
    
    def update_weights(self):
        for layer in self.layers:
            layer.weights += layer.dW
            layer.bias += layer.db

    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        self.clear_grad()
        n_samples = len(X)
        for i in range(n_samples):
            y_pred = self.forward(X[i])
            self.backpropagation(y_pred, y[i])
        
        # 更新权重和阈值
        for layer in self.layers:
            layer.dW /= n_samples
            layer.db /= n_samples
        self.update_weights()
    
    def predict(self, X):
        # 对样本进行预测
        return np.array([self.forward(x) for x in X])
