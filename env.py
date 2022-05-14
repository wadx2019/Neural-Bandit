import numpy as np
import pandas as pd
import random as rd

class IRIS():
    def __init__(self):
        self.arm = 3
        self.dim = 12
        self.data = pd.read_csv('Iris.csv')

    def step(self):
        r = rd.randint( 0, 149)
        if  0 <= r <= 49:
            target = 0
        elif 50 <= r <= 99:
            target = 1
        else:
            target = 2
        random = self.data.loc[r]
        x = np.zeros(4)
        for i in range(1,5):
            x[i-1] = random[i]
        X_n = []
        for i in range(3):
            front = np.zeros((4 * i))
            back = np.zeros((4 * (2 - i)))
            new_d = np.concatenate((front, x, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)
        reward = np.zeros(self.arm)
        # print(target)
        reward[target] = 1
        return X_n, reward
