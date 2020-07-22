# -*- coding: utf-8 -*

from sklearn import metrics
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


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


def main():
    samples = make_classification(n_samples=1000,
                                  n_features=50,
                                  n_classes=2,
                                  random_state=1,
                                  weights=[0.15],
                                  flip_y=0.1
                                  )
    samples_train_x = samples[0][:-100]
    samples_train_y = samples[1][:-100]
    samples_test_x = samples[0][-100:]
    samples_test_y = samples[1][-100:]

    KNN_clf = KNeighborsClassifier(n_neighbors=2)
    KNN_clf.fit(samples_train_x, samples_train_y)

    SVM_clf = svm.SVC(probability=True)
    SVM_clf.fit(samples_train_x, samples_train_y)

    score = KNN_clf.score(samples_test_x, samples_test_y)
    print("KNN 模型打分：Score = %f" % score)
    KNN_y_predict = KNN_clf.predict(samples_test_x)
    accuracy = Get_Accuracy(samples_test_y, KNN_y_predict)
    print("KNN Accuracy_Score = %f" % accuracy)
    precision = Get_Precision(samples_test_y, KNN_y_predict)
    print("KNN Precision_Score = %f" % precision)
    recall = Get_Recall(samples_test_y, KNN_y_predict)
    print("KNN Recall_Score = %f" % recall)
    f1_score = Get_F1(samples_test_y, KNN_y_predict)
    print("KNN F1-Score  = %f" % f1_score)

    print("________________分割线________________")

    score_S = SVM_clf.score(samples_test_x, samples_test_y)
    print("SVM 模型打分: Score = %f" % score_S)
    SVM_y_predict = SVM_clf.predict(samples_test_x)
    accuracy_S = Get_Accuracy(samples_test_y, SVM_y_predict)
    print("SVM Accuracy_Score = %f" % accuracy_S)
    precision_S = Get_Precision(samples_test_y, SVM_y_predict)
    print("SVM Precision = %f" % precision_S)
    recall_S = Get_Recall(samples_test_y, SVM_y_predict)
    print("SVM Recall = %f" % recall_S)
    f1_score_S = Get_F1(samples_test_y, SVM_y_predict)
    print("SVM F1-Score  = %f" % f1_score_S)


main()
