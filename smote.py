# SMOTE Method
import random
import numpy as np


class SMOTE:
    def __init__(self, k=2, m=0.5, r=2):
        self.k = k  # Number of neighbors: [1,20]
        # Number of synthetic examples to create.Expressed as percent of final training data
        self.m = m  # [50, 100, 200, 400]
        self.r = r  # Power parameter for the Minkowski distance metric: [0.1, 5]

    def fit_sample(self, data, label):
        # data : 包含度量信息的样本 数组
        # label : 样本的标签 数组
        data_t, data_f, label_t, label_f = [], [], [], []
        # 按照正例和反例划分数据集
        N = label.shape[0]  # 样例总数
        for i in range(N):
            if label[i] == 1:
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])
        T = int(self.m /100 * N)  # 每一类需要的样例数
        num_minority = len(data_t)
        if self.k >= num_minority:
            # 如果k大于少数类个数，重制K，避免找不到足够多的邻居
            self.k = num_minority - 1
        # 剔除多余负类样本
        while len(label_f) > T and len(label_f) != 0:
            remove_index = random.randrange(0, len(label_f))
            data_f.pop(remove_index)
            label_f.pop(remove_index)
        # 生成新样本
        new_sample_list, new_sample_num = [], 0
        while new_sample_num < T - len(label_t):
            current_index = random.randrange(0, len(label_t))
            new_sample_list.extend(self.something_like(data_t, data_t[current_index]))
            new_sample_num += self.k
        data_t.extend(new_sample_list)
        label_new = np.ones(len(new_sample_list))
        label_t.extend(label_new)
        return np.append(data_t, data_f, axis=0), np.append(label_t, label_f, axis=0)

    def something_like(self, data, x0):
        relevant = []
        k1 = 0
        neighbors = self.found(data, x0)
        for neighbor in neighbors:
            # 计算得到新样例的向量坐标
            bar_ab = neighbor - x0
            bar_ac = random.random() * bar_ab
            c = x0 + bar_ac
            relevant.append(c)
        return relevant
    # 闵科夫斯基距离
    def minkowski_distance(self, a, b):
        distance_r = 0
        for i in range(len(a)):
            distance_r += pow(abs(a[i]-b[i]), self.r)
        return pow(distance_r, 1/self.r)

    # 找到邻居点
    def found(self, data, x0):
        distances = []
        for i in range(len(data)):
            d = self.minkowski_distance(x0, data[i])
            d_tuple = (i, d)
            distances.append(d_tuple)
        distances.sort(key=lambda x:x[1])  # 升序排序
        return [data[distances[i][0]] for i in range(1, int(self.k+1))]

