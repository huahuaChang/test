# -*- coding: utf-8 -*

import io
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.datasets import make_classification

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def Get_Accuracy(y_true, y_pred):  # Accuracy-精准率：分类器正确分类的样本与总样本数之比
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy


def Get_Precision(y_true, y_pred):  # Precision-精准率：被正确预测的正样本/所有被预测为正样本的比例
    precision = metrics.precision_score(y_true, y_pred)
    return precision


def Get_Recall(y_true, y_pred):  # Recall-召回率：被正确预测为正样本/所有真正正样本
    recall = metrics.recall_score(y_true, y_pred)
    return recall


def Get_F1(y_true, y_pred):  # F1-F分数：precision和recall的调和平均数
    F1 = metrics.f1_score(y_true, y_pred)
    return F1


def Get_AUC(y_true, y_probility):
    auc = metrics.roc_auc_score(y_true, y_probility)
    return auc


def main():
    # 随机生成数据集
    samples = make_classification(n_samples=1000,  # 1000个样本
                                  n_features=50,  # 每个样本包含50个特征
                                  n_classes=2,  # 两类
                                  random_state=1,  # 随机种子，保证稳定
                                  weights=[0.15],  # 权重
                                  flip_y=0.1
                                  )
    # print(samples)
    # df = pd.DataFrame(samples[0], samples[1])
    # df.to_csv("samples.csv")
    # 截取前900个样本作为训练集，后100个作为测试集
    samples_train_x = samples[0][:-100]
    samples_train_y = samples[1][:-100]
    samples_test_x = samples[0][-100:]
    samples_test_y = samples[1][-100:]

    # 初始化一个svm并训练
    clf = svm.SVC(probability=True)
    # clf = svm.SVC(gamma='auto')
    clf.fit(samples_train_x, samples_train_y)

    # 用训练好的模型对trainsets进行测试
    y_predict = clf.predict(samples_test_x)
    # print("SVM测试集预测结果：")
    # print(y_predict)

    y_predict_probability = clf.predict_proba(samples_test_x)
    print(y_predict_probability)

    # AUC用到
    df2 = pd.DataFrame(y_predict_probability)
    proba_pred_y = np.array(df2[1])  # 获取样本点预测为正样本的预测概率
    # df2.to_csv("pred_probability.csv")
    # print(proba_pred_y)

    score = clf.score(samples_test_x, samples_test_y)
    print("SVM 模型打分: Score = %f" % score)
    accuracy = Get_Accuracy(samples_test_y, y_predict)
    print("SVM Accuracy_Score = %f" % accuracy)
    precision = Get_Precision(samples_test_y, y_predict)
    print("SVM Precision = %f" % precision)
    recall = Get_Recall(samples_test_y, y_predict)
    print("SVM Recall = %f" % recall)
    f1_score = Get_F1(samples_test_y, y_predict)
    print("SVM F1-Score  = %f" % f1_score)
    auc = Get_AUC(samples_test_y, proba_pred_y)
    print("SVM AUC value: AUC = %f" % auc)


main()
