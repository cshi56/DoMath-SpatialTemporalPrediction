import numpy as np
import random
import math
import matplotlib.pyplot as plt

random.seed(1234)
np.random.seed(1234)


class Node:
    def __init__(self, alpha, beta, gamma, n, s, e, i):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n = n

        self.s = s
        self.e = e
        self.i = i
        self.r = n - s - e - i

        self.s_to_e_coefficient = beta * i * s / n
        self.e_to_i_coefficient = alpha * e
        self.i_to_r_coefficient = gamma * i

        self.total_time = 0.0
        self.history = []

    def update_coefficients(self):
        self.s_to_e_coefficient = self.beta * self.i * self.s / self.n
        self.e_to_i_coefficient = self.alpha * self.e
        self.i_to_r_coefficient = self.gamma * self.i


class Simulation:
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
