# auto SMOTE Method
import numpy as np
import random
from SMOTUNED.smote import SMOTE
from sklearn.linear_model import LogisticRegression


class SMOTUNED(object):
    def __init__(self, n=10, cf=0.3, f=0.7):
        self.n = n  # Population Size: Frontier size in a generation
        self.cf = cf  # Crossover probability: Survival of the candidate
        self.f = f  # Differential weight: Mutation power
        self.k = 20  # Number of neighbors: [1,20]
        # Number of synthetic examples to create.Expressed as percent of final training data
        self.m = [50, 100, 200, 400]
        self.r = 5  # Power parameter for the Minkowski distance metric: [0.1, 5]

    def DE(self, data, label):
        frontier = self.guessed()
        best = frontier[0]
        lives = 1  # Number of generations
        while lives > 0:
            lives -= 1
            tmp = []
            for i in range(len(frontier)):
                old = frontier[i]
                index = random.sample(range(len(frontier)), 3)
                x, y, z = frontier[index[0]], frontier[index[1]], frontier[index[2]]
                new = old.copy()
                for j in range(len(new)):
                    if random.random() < self.cf:
                        if j ==0:
                            temp_1 = int(x[j] + self.f * (z[j] - y[j]))
                            if temp_1 >= 1 and temp_1 <= 20:
                                new[j] = temp_1
                        if j ==1:
                            temp_2 = x[j] + self.f * (z[j] - y[j])
                            if temp_2 > 0:
                                new[j] = temp_2
                        if j ==2:
                            temp_3 = round(x[j] + self.f * (z[j] - y[j]), 1)
                            if temp_3 >= 0.1 and temp_3 <= 5:
                                new[j] = temp_3
                new, is_ = self.better(new, old, data, label)
                tmp.append(new)
                t, is_ = self.better(new, best, data, label)
                if is_:
                    best = t
                    lives += 1
            frontier = tmp
        return best[0], best[1], best[2]


    # 初始化frontier
    def guessed(self):
        frontier = []
        for i in range(self.n):
            k = random.randint(1, self.k+1)
            m = random.choice(self.m)
            r = random.randint(1, self.r*10+1)/10
            frontier.append([k, m, r])
        return frontier

    @staticmethod
    # fitness Function
    def better(new, old, data, label):
        data_new, label_new = SMOTE(k=new[0], m=new[1], r=new[2]).fit_sample(data, label)
        data_old, label_old = SMOTE(k=old[0], m=old[1], r=old[2]).fit_sample(data, label)
        clf_new = LogisticRegression()
        clf_new.fit(data_new, label_new)
        score_new = clf_new.score(data_new, label_new)
        clf_old = LogisticRegression()
        clf_old.fit(data_old, label_old)
        score_old = clf_old.score(data_old, label_old)
        if score_new > score_old:
            return new, True
        else:
            return old, False
