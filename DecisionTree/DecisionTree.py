# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:58:32 2020

@author: Dragonfly
@email: liuxiangmin@tom.com
"""

# 使用ID3算法做分类
# 使用信息增益准则分裂子节点。信息熵：sum(-plogp)

# 使用Mnist数据。训练数据60000个样本，测试数据10000个样本
# accuracy: 0.8636
# time: 703s

import pandas as pd
import numpy as np
from math import log
import time
from collections import Counter

def loadData(fileName):
    """
    从文件读取数据，并做一定处理
    param fileName: 数据文件路径
    return: 训练特征，对应的标签
    """
    data=pd.read_csv(fileName,header=None)
    data=data.values
    #数据第一行为分类结果
    label = data[:,0]
    data_X = data[:,1:]

    # 因为data_X的取值范围为0-255，则分裂子节点时可能性过多，计算过于繁杂，做二值化处理。
    data_X[data_X<128]=0
    data_X[data_X>=128]=1

    return data_X, label

def entropy(label):
    """
    计算熵
    param label: 1维数据
    return: float值
    """
    label_counts = {}
    for i in range(len(label)):
        if label[i] not in label_counts.keys():
            label_counts[label[i]] = 0
        label_counts[label[i]] += 1
    ent = - sum([(p/len(label)) * log(p/len(label), 2) for p in label_counts.values()])
    return ent

def con_entropy(data, label, col):
    """
    计算条件熵
    param data: numpy array特征数据
    param laebel: 1维标签数据
    param col: 用于对数据做划分的特征列序号
    return: float值
    """
    feature_sets = {}
    for i in range(len(data)):
        feature = data[i][col]
        if feature not in feature_sets.keys():
            feature_sets[feature] = []
        feature_sets[feature].append(data[i])
    con_ent = sum([(len(label[data[:, col]==p])/len(data)) * entropy(label[data[:, col]==p]) for p in feature_sets.keys()])
    return con_ent

def find_best_feature(data, label):
    """
    寻找信息增益最大的特征
    param data: numpy array数据
    return: （信息增益最大特征所在列序号，最大信息增益值）
    """
    ent = entropy(label)
    info_gains = []
    for col in range(data.shape[1]):
        con_ent = con_entropy(data, label, col)
        info_gains.append(ent - con_ent)
    print(max(info_gains))
    return info_gains.index(max(info_gains)), max(info_gains)

def cut_data(data, label, Ag, ai):
    """
    按最优特征对数据进行切分
    param data: 待切分特征数据
    param laebel: 1维标签数据
    param Ag: 最优切分特征
    param ai: 最优切分特征的其中一个特征值
    return: 切分后，特征Ag的值为ai的数据和相应标签
    """
    part_data = []
    part_label = []
    for i in range(len(data)):
        if data[i][Ag] == ai:
            part_data.append(list(data[i][0:Ag]) + list(data[i][Ag+1:]))
            part_label.append(label[i])
    return np.array(part_data), np.array(part_label)

def find_class(label):
    """
    确定树节点的类别
    param label: 1维标签数据
    return: 标签/类别
    """
    counter = Counter(label)
    return counter.most_common(1)[0][0]

def dt_train(data, label, epsilon=0.1):
    """
    训练决策树
    param data: 训练特征数据
    param label: 训练标签数据
    param epsilon: 信息增益的阈值
    return: 使用字典描述的决策树
    """
    print("Create node, %d label data to split..." % len(label))

    clusters = set([i for i in label])  # 查看还有多少分类

    # 如果所有实例属于同一类，则决策树T为单节点树，返回该类作为该节点的类标记
    if len(clusters) == 1:
        return label[0]

    # 如果可选特征为空集，则决策树T为单节点树，返回实例数最大的类作为该节点的类标记
    if len(data) == 0:
        return find_class(label)

    # 计算最大信息增益特征及最大信息增益值
    Ag, max_info_gain = find_best_feature(data, label)

    # 如果最大信息增益小于阈值，则决策树T为单节点树，返回实例树最大的类作为该节点的类标记
    if max_info_gain < epsilon:
        return find_class(label)

    # 使用字典描述树，如tree{378:{0：{},1: {}}
    # 就代表按第387列特征分裂节点，按该列特征值取0、还是1分裂为2棵子树，各个子树又可以构造子树
    tree = {Ag: {}}

    # 按特征取值，对数据进行切分
    for ai in set([i for i in data[:, Ag]]):
        part_data, part_label = cut_data(data, label, Ag, ai)
        tree[Ag][ai] = dt_train(part_data, part_label)

    return tree

def dt_predict(row, tree):
    """
    使用训练好的决策树逐行预测
    param row: 待预测的特征数据,list数据
    param tree: 训练好的决策树
    return: 预测的类别
    """
    while True:
        (key, value), = tree.items()
        if type(value).__name__ == 'dict':        # 如果是字典，说明还没到叶子节点，继续搜索子树
            feature = row[key]                    # 获取当前分裂特征下，row对应的特征值
            del row[key]                          # 将已经考虑的特征剔除
            tree = value[feature]                 # 用子树更新tree的值
            if type(tree).__name__ == "int64":
                return tree
        else:
            return value

def accuracy(y, y_pred):
    count = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            count += 1
    return count/len(y)

if __name__ == "__main__":
    # 获取开始时间
    start = time.time()
    # 读取训练测试数据
    X_train, y_train = loadData('../Mnist_data/train.csv')
    X_test, y_test = loadData('../Mnist_data/test.csv')

    # 训练决策树
    tree = dt_train(X_train, y_train)
    # 利用决策树做预测
    pred_y = []
    for i in range(len(X_test)):
        pred_y.append(dt_predict(list(X_test[i]), tree))

    # 模型性能评估
    score = accuracy(y_test, pred_y)
    print(score)
    # 获取结束时间
    end = time.time()
    print("run time: ", end - start)
