# -*- coding: utf-8 -*

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.color = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training
    def fit(self, data):
        self.data = data
        opt_dict = {}  # 键值对类型 pair key:value  {||w||:[w,b]}

        # 从x的正半轴到负半轴转180°，按照指定的步长（）来构造transforms矩阵
        rotMatrix = lambda theta: np.array([np.sin(theta), np.cos(theta)])
        # cy = ax+b1 -> y = (ax+b1)/c = ax/c+b sin/cos = tanTheta = a/c
        thetaStep = np.pi / 10
        transforms = [np.array(rotMatrix(theta))
                      for theta in np.arange(0, np.pi, thetaStep)]
        # 将数据集从二维变成一个list
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        # 找到数据集中的最大、最小值
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # 定义步长
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        # 计算b和w
        b_rang_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_rang_multiple),
                                   self.max_feature_value * b_rang_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # latest_optimum = opt_choice[0][0] * step * 2
            # latest_optimum = opt_choice[0][0] * step

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.color[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.color[i]) for x in data_dict[i]] for i in data_dict]

        # line = x.w+b  cy=ax+b w->w[0]=a, w[1]=c
        def hyperplane(x, w, b, y):
            rst = (-w[0] * x - b + y) / w[1]
            return rst

        data_range = (self.min_feature_value, self.max_feature_value)

        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 8],
                           [2, 3],
                           [3, 6]], ),
             1: np.array([[1, -2],
                          [3, -4],
                          [3, 0]])}


def main():
    # 初始化一个svm
    svm = Support_Vector_Machine()
    # 使用初始数据训练一个分类器
    svm.fit(data=data_dict)

    svm.visualize()


main()
