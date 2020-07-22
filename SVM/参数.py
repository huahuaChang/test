
#-*- coding:utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


# 总共 40 个人，每人 10 幅图像
def get_data():
    data = np.empty([400, 10304])
    target = np.empty(400)
    ad = './ORL/'
    num = 0
    for file1 in os.listdir(ad):
        for i in range(1, 11):
            im = Image.open(ad + file1 + '/' + str(i) + '.bmp')
            data[num] = np.array(im.getdata())
            target[num] = int(file1[1:])
            num += 1
    return data, target


# 画图
def get_picture(comps, y, kernel):
    plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.ylabel('正确率')
    plt.xlabel('降维后的维数')
    plt.title('不同参数下SVM人脸识别折线图')
    for i in range(len(y)):
        plt.scatter(comps, y[i], s=30, c='r', marker='x', linewidths=1)
        plt.plot(comps, y[i], label=kernel[i] + "核函数")
    plt.legend()
    plt.show()


def svm_k(X, y, fx):
    ans = 0
    kf = KFold(n_splits=10, shuffle=True)
    for train, test in kf.split(X):
        clf = SVC(kernel=fx, gamma='auto').fit(X[train], y[train])
        ans += clf.score(X[test], y[test])
    return ans / 10


def svm(X, y, fx):
    scores = cross_val_score(SVC(kernel=fx, gamma='auto'),
                             X, y, cv=10)
    return scores.mean()


if __name__ == "__main__":
    X_data, y_data = get_data()

    # 归一化
    scaler = MinMaxScaler().fit(X_data)
    X_data_s = scaler.transform(X_data)

    comps = [20, 50, 70, 100, 130, 150, 170, 200]  # 降维到 20,50,100,200
    kernel = ['rbf', 'linear', 'poly', 'sigmoid']  # 核函数不同
    y = np.empty([len(kernel), len(comps)])  # 存储不同维度的精度
    for i in range(len(comps)):
        print('第{}维开始'.format(comps[i]))
        # PCA降维
        pca = PCA(n_components=comps[i]).fit(X_data_s)
        X_data_pca = pca.transform(X_data_s)
        # SVM分类
        for j in range(len(kernel)):
            y[j, i] = svm(X_data_pca, y_data, kernel[j])

    get_picture(comps, y, kernel)
