# -*- coding: utf-8 -*

import numpy as np
from sklearn.datasets import load_digits  # 手写数字数据集
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # 训练和测试集分隔
from sklearn.preprocessing import LabelBinarizer  # 标签二值化处理

# 载入数据
#使用sklearn库中的数字集的数据。它是由1797张手写数字图片组成，图片的像素是8x8.
digits = load_digits()
print(digits.images.shape)  # 结果：(1797, 8, 8)

# 输入的数据
X = digits.data
# 标签数据
T = digits.target

print(X.shape, X[:2], '\n')
print(T.shape, T[:2])
# 结果：
# (1797, 64) [[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#   15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#    0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#    0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
#  [ 0.  0.  0. 12. 13.  5.  0.  0.  0.  0.  0. 11. 16.  9.  0.  0.  0.  0.
#    3. 15. 16.  6.  0.  0.  0.  7. 15. 16. 16.  2.  0.  0.  0.  0.  1. 16.
#   16.  3.  0.  0.  0.  0.  1. 16. 16.  6.  0.  0.  0.  0.  1. 16. 16.  6.
#    0.  0.  0.  0.  0. 11. 16. 10.  0.  0.]]

# (1797,) [0 1]

# 定义一个神经网络 64-100-10，隐藏层神经元单元为100,输出层神经元单元为10
# 输入层至隐藏层的权值矩阵
V = np.random.random([64, 100]) * 2 - 1
# 隐藏层至输出层的权值矩阵
W = np.random.random([100, 10]) * 2 - 1

# 数据切分，默认测试集占0.25
X_train, X_test, y_train, y_test = train_test_split(X, T)

# 标签二值化，独热编码
# 1 -> 0100000000
labels_train = LabelBinarizer().fit_transform(y_train)
print(y_train[:2])
print(labels_train[:2])


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激活函数的导数
def dsigmoid(x):
    return x * (1 - x)


# 预测值计算
def predict(x):
    L1 = sigmoid(np.dot(x, V))
    L2 = sigmoid(np.dot(L1, W))
    return L2


# 模型训练
def train(X, T, steps=10000, lr=0.11):
    global V, W
    for n in range(steps + 1):
        # 从样本中随机选取一个数据
        i = np.random.randint(X.shape[0])
        x = X[i]
        x = np.atleast_2d(x)  # 转换为2D矩阵
        # BP算法公式
        L1 = sigmoid(np.dot(x, V))
        L2 = sigmoid(np.dot(L1, W))

        L2_delta = (T[i] - L2) * dsigmoid(L2)
        L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)

        W += lr * L1.T.dot(L2_delta)
        V += lr * x.T.dot(L1_delta)

        if n % 1000 == 0:
            output = predict(X_test)
            predictions = np.argmax(output, axis=1)
            acc = np.mean(np.equal(predictions, y_test))
            print('iter: ' + str(n + 1) + " acc: " + str(acc))


train(X_train, labels_train, 30000)


# 结果评估
output = predict(X_test)
predictions = np.argmax(output, axis=1)
print(classification_report(predictions, y_test))


# 混淆矩阵
print(confusion_matrix(predictions, y_test))


