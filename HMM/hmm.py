"""
Created on Tue Jul 28 15:05:25 2020

@author: liuxiangmin
@email: liuxiangmin@tom.com
"""

# HMM模型是关于时序的概率模型，描述由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测从而产生观测随机序列的过程。
# HMM的齐次马尔科夫性假设，即当前状态只和前一个状态有关，与其他时刻的状态及观测无关。
# HMM的观测独立性假设，即当前观测只和当前状态有关，与其他时刻的状态及观测无关。
# HMM模型由状态转移概率矩阵A，观测概率矩阵B，初始状态向量概率pi，描述。

# 训练数据：已经分词过的人民日报1998语料库
# 模型学习：（A, B, pi）的参数估计。训练数据已经分词完毕，使用极大似然估计。
# 模型预测：使用维特比算法。
# 模型评估：使用精度precision, 召回率recall。

# 选用网上找的近期一段人民日报语料做测试，精度85%，召回率89%。

import numpy as np
from tqdm import tqdm

# 在中文分词中，包含以下几种状态（词性)
# B: 词语的开头
# M: 中间词
# E: 词语的结尾
# S: 孤立的单个字
# 定义一个状态映射字典。方便我们定位状态在列表中对应位置
status2num = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

# 定义状态转移概率矩阵。总共4个状态，所以4x4
A = np.zeros((4, 4))

# 定义观测概率矩阵
# 使用python内置的ord函数获取词的编码，编码大小为65536，总共4个状态
# 所以B矩阵4x65536
B = np.zeros((4, 65536))

# 初始状态，每一个句子的开头只有4种状态（词性）
pi = np.zeros(4)

# 求HMM的参数（A, B, pi)
def hmm_train(fileName):
    """
    :param fileName: 用于训练hmm参数的语料文件
    :return: （A, B, pi)
    """
    with open(fileName, encoding='utf-8') as file:  # 读取语料文件，需根据情况修改文件路径
        lines = file.readlines()  # 返回list

    for line in tqdm(lines):
        line = line.strip().split()
        all_words_status = []   # 记录每一行所有词的状态
        for words in line:         # 计算B矩阵
            status = []  # 记录每一分词的状态
            if len(words) == 1:
                status.extend("S")
            else:
                status.extend("B" + "M" * (len(words) - 2) + "E")
            all_words_status.extend(status)   # 一行的状态，供计算A, pi时使用
            for i in range(len(words)):
                B[status2num[status[i]]][ord(words[i])] += 1

        # 计算A矩阵
        for i in range(len(all_words_status)-1):
            A[status2num[all_words_status[i]]][status2num[all_words_status[i + 1]]] += 1

        # 计算pi向量
        if len(all_words_status) != 0:  # 跳过没有内容的行，不然会报错
            pi[status2num[all_words_status[0]]] += 1  # 每一行的第一个字的状态为初始状态

    # 转换为概率
    # 如果句子较长，许多个较小的概率值连乘，容易造成下溢。对于这种情况，可以使用log函数解决。
    # 但是当碰到0时，log0是没有定义的，我们给每一个0的位置加上一个极小值-3.14e+100，使其有定义。
    for i in range(len(A)):
        row_sum = np.sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0:
                A[i][j] = -3.14e+100
            else:
                A[i][j] = np.log(A[i][j] / row_sum)

    for i in range(len(B)):
        row_sum = np.sum(B[i])
        for j in range(len(B[i])):
            if B[i][j] == 0:
                B[i][j] = -3.14e+100
            else:
                B[i][j] = np.log(B[i][j] / row_sum)

    pi_sum = np.sum(pi)
    for i in range(len(pi)):
        if pi[i] == 0:
            pi[i] = -3.14e+100
        else:
            pi[i] = np.log(pi[i] / pi_sum)

    return A, B, pi

# hmm模型参数训练好后，使用维特比算法做分词
def hmm_predict(article, hmm_param):
    """
    使用hmm训练好的参数及维特比算法做分词
    param article: 待分词的文字
    param hmm_param: hmm参数，(A, B, pi)
    param return: 分词后的文字
    """
    A, B, pi = hmm_param
    article_partition = []          # 保存分词后的结果
    for line in article:
        line = line.strip()
        # 维特比算法
        # delta--长度为每一行长度，每一位有4种状态
        delta = [[0 for _ in range(4)] for _ in range(len(line))]
        # psi同理
        psi = [[0 for _ in range(4)] for _ in range(len(line))]
        psi[0][:] = [0, 0, 0, 0]
        for i in range(4):
            delta[0][i] = pi[i] + B[i][ord(line[0])]  # 求初始时刻的delta。psi用零初始化即可，无需再求。
        for t in range(1, len(line)):
            for i in range(4):
                tmp = [delta[t - 1][j] + A[j][i] for j in range(4)]  # 求t-1时刻状态转变为t时刻状态的所有可能概率取值
                delta[t][i] = max(tmp) + B[i][ord(line[t])]
                psi[t][i] = tmp.index(max(tmp))

        status = []  # 保存最优状态链
        It = delta[-1].index(max(delta[-1]))  # 已求出的最新一个时刻的状态
        status.append(It)
        for t in range(len(line) - 1, 0, -1):
            status.insert(0, psi[t][It])
            It = psi[t][It]  # 更新It

        # 根据状态做分词
        partition_line = ""  # 保存分词后的行结果
        for i in range(len(line)):
            partition_line += line[i]
            if (status[i] == 2 or status[i] == 3) and (i != len(line) - 1):  # 如果字的状态为E或S，且不在行尾，则在末尾加空格。
                partition_line += " "
        article_partition.append(partition_line)
    return article_partition

# 评估hmm模型分词效果
def to_region(segmentation):
    """
    将一行分词结果转换为区间
    param segmentation: 百年 未有 之 大变局
    return: [(0, 2), (2, 4), (4, 5), (5, 8)]
    """
    sequence = segmentation.split()
    result = []
    count0 = 0
    count1 = 0
    for words in sequence:
        count1 += len(words)
        result.append((count0, count1))
        count0 = count1
    return result

def hmm_performance(perfect, pred):
    """
    计算分词的精度和召回率
    param perfect: 标准分词结果
    param pred: 算法分词结果
    param return: (precision, recall)
    """
    A_size, B_size, A_cap_B_size = 0, 0, 0
    A, B = set(to_region(perfect)), set(to_region(pred))
    A_cap_B = A & B
    A_size += len(A)
    B_size += len(B)
    A_cap_B_size += len(A_cap_B)
    return A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100

if __name__ == '__main__':
    A, B, pi = hmm_train("RMRB_yuliao.txt")     # 训练hmm
    # 读取测试语料
    with open("test.txt", encoding='utf-8') as f:
        article = f.readlines()
    partition_article = hmm_predict(article, (A, B, pi))
    # 读取手工分词的语料
    with open("test_perfect.txt", encoding='utf-8') as f:
        perfect_partition_article = f.readlines()

    # 将分词语料整合为一行
    perfect_article = ""
    for words in perfect_partition_article:
        perfect_article += words + " "
    perfect_article = perfect_article.strip()

    pred_article = ""
    for words in partition_article:
        pred_article += words + " "
    pred_article = pred_article.strip()

    # 计算hmm分词效果
    hmm_precision, hmm_recall = hmm_performance(perfect_article, pred_article)
    print("hmm precision: %f" % hmm_precision)
    print("hmm recall: %f" % hmm_recall)
