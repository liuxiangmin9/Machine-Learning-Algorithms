# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:46:02 2020

@author: liuxiangmin
@email: liuxiangmin@tom.com
"""

# 逻辑斯蒂回归模型可用于二分类或多分类。二分类时，p(y=1|x)=sigmoid(x)=1/(1+exp(-w*x)),p(y=0|x)=1-p(y=1|x)
# 可以用极大似然估计法估计模型权重系数。

# 使用IMDB电影评论数据，做one-hot编码。训练数据和测试数据各25000个样本
# accuracy: 0.849

from math import exp
from tqdm import tqdm
import numpy as np
from tensorflow.keras.datasets import imdb

# LR模型构建
class LogisticRegressionClassifier:
    def __init__(self, max_epochs=10, learning_rate=0.01):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    # 定义逻辑斯蒂分布的分布函数,即sigmoid函数
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # 给各样本加上特征“1”，以使截距b包含在w中
    def data_matrix(self, X):
        data_mat = []
        for d in tqdm(X):
            data_mat.append([1.0, *d])
        return data_mat

    # 拟合模型
    def fit(self, X, y):
        data_mat = self.data_matrix(X)
        # 定义权重w
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)
        # 迭代max_epochs次，使用单个样本对权重做梯度上升更新
        for epoch in tqdm(range(self.max_epochs)):
            for i in range(len(data_mat)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])  # 似然函数的梯度

    # 模型预测
    def predict(self, x_test):
        data_mat = self.data_matrix(x_test)
        result = []
        for i in range(len(data_mat)):
            temp = np.dot(data_mat[i], self.weights)
            if temp >= 0:
                result.append(1)
            else:
                result.append(0)
        return result

    # 模型评估
    def accuracy(self, y, y_pred):
        right = 0
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                right += 1
        return right / len(y)

if __name__ == "__main__":
    # 引入IMDB电影评论数据做二分类
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)  # 保留训练数据中前1000 个最常出现的单词

    # 填充列表，使其具有相同的长度，做one-hot编码
    def vectorize_sequences(sequences, dimension=1000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # 调用LR类
    LR = LogisticRegressionClassifier()
    # 训练模型
    LR.fit(x_train, train_labels)
    # 模型预测
    test_pred = LR.predict(x_test)
    # 模型评估
    score = LR.accuracy(test_labels, test_pred)
    print(score)
