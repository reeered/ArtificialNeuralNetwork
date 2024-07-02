import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ann.activation import relu
from ann.layer import Layer
from ann.ann import ANN

diabetes = load_diabetes()
train_X, test_X, train_y, test_y = train_test_split(diabetes.data, diabetes.target, test_size=0.3)

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean((y_true - y_pred) ** 2)

ann = ANN([Layer(10, 18), Layer(18, 9), Layer(9, 1, activation=relu)], lr=0.001)
epochs = 1500
best = 1e9
mse_train_list = []
mse_test_list = []
for i in range(epochs):
    if i % 100 == 0:
        ann.lr *= 0.8
        if i != 0 and i % 500 == 0:
            ann.lr = 0.001 + 0.001 * i / 500
    ann.fit(train_X, train_y)

    y_pred_train = ann.predict(train_X)
    mse_train = mse_loss(train_y, y_pred_train)

    y_pred_test = ann.predict(test_X)
    mse_test = mse_loss(test_y, y_pred_test)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    
    if mse_test < best:
        best = mse_test
        best_epoch = i

    print(f'epoch {i}, 训练集mse: {mse_train}, 测试集mse: {mse_test}')
    
print(f'best epoch: {best_epoch}, mse: {best}')

# 使用sklearn的MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(18, 9), max_iter=10000)
mlp.fit(train_X, train_y)
y_pred = mlp.predict(test_X)
print("sklearn MLPRegressor 测试集mse: ", mse_loss(test_y, y_pred))

# 使用sklearn的LinearRegression
lr = LinearRegression()
lr.fit(train_X, train_y)
y_pred = lr.predict(test_X)
print("sklearn LinearRegression 测试集mse: ", mse_loss(test_y, y_pred))

# 绘制mse变化曲线
plt.figure()
plt.title('mse on train and test')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.plot(mse_train_list, label='train', color='r')
plt.plot(mse_test_list, label='test', color='b')
plt.legend()
plt.show()
