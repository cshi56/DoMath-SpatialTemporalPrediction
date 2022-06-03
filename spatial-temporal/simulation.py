import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1234)
np.random.seed(1234)
np.set_printoptions(threshold=np.inf)


def my_log(number):
    if number == 0:
        return -1 * np.infty
    return np.log(number)


def my_divide(a, b):
    if b == 0:
        return np.sign(a) * np.infty
    return a / b


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
        self.history = [[self.total_time, self.s, self.e, self.i, self.r]]

        self.next_unit_time_step = 1
        self.unit_time_history = [[0, self.s, self.e, self.i, self.r]]

    def update_coefficients(self):
        self.s_to_e_coefficient = self.beta * self.i * self.s / self.n
        self.e_to_i_coefficient = self.alpha * self.e
        self.i_to_r_coefficient = self.gamma * self.i

    def update_seir(self, argmin):
        if argmin == 0:
            self.s -= 1
            self.e += 1
        elif argmin == 1:
            self.e -= 1
            self.i += 1
        else:
            self.i -= 1
            self.r += 1

    def time_step(self):
        self.update_coefficients()

        if max(self.s_to_e_coefficient, self.e_to_i_coefficient, self.i_to_r_coefficient) == 0:
            self.total_time += 0.1
            self.history.append([self.total_time, self.s, self.e, self.i, self.r])
            return

        s_to_e_time = -1 * my_divide(my_log(random.uniform(0, 1)), self.s_to_e_coefficient)
        e_to_i_time = -1 * my_divide(my_log(random.uniform(0, 1)), self.e_to_i_coefficient)
        i_to_r_time = -1 * my_divide(my_log(random.uniform(0, 1)), self.i_to_r_coefficient)

        time_array = np.asarray([s_to_e_time, e_to_i_time, i_to_r_time])
        self.total_time += min(time_array)
        argmin = np.argmin(time_array)
        self.update_seir(argmin)
        self.history.append([self.total_time, self.s, self.e, self.i, self.r])

    def graph(self):
        history = np.asarray(self.unit_time_history)

        time_data = history[:, 0]
        s_data = history[:, 1]
        e_data = history[:, 2]
        i_data = history[:, 3]
        r_data = history[:, 4]

        plt.plot(time_data, s_data, label='Susceptible subjects')
        plt.plot(time_data, e_data, label='Exposed subjects')
        plt.plot(time_data, i_data, label='Infected subjects')
        plt.plot(time_data, r_data, label='Removed subjects')
        plt.legend()
        plt.show()

    def simulate_single_time_unit(self):
        while self.total_time <= self.next_unit_time_step:
            self.time_step()

        self.history.pop()
        self.total_time = self.next_unit_time_step

        unit_data_to_add = [self.next_unit_time_step]
        unit_data_to_add.extend(self.history[-1][1:])

        self.unit_time_history.append(unit_data_to_add)
        self.history.append(unit_data_to_add)  # can do this due to lack of memory of Poisson processes
        self.s, self.e, self.i, self.r = \
            self.history[-1][1], self.history[-1][2], self.history[-1][3], self.history[-1][4]
        self.next_unit_time_step += 1

    def simulate(self):
        while self.unit_time_history[-1][2] + self.unit_time_history[-1][3] != 0:
            self.simulate_single_time_unit()


class Simulation:
    def __init__(self):
        self.nodes = []
        self.diffusion_matrix = []

    def add_node(self, node):
        self.nodes.append(node)
        # something ab diffusion matrix

    def simulate_single_time_unit(self):
        for node in self.nodes:
            node.simulate_single_time_unit()


if __name__ == '__main__':
    ALPHA = 0.1
    BETA = 0.4
    GAMMA = 0.05
    N = 500000
    S = 499900
    E = 0
    I = 100

    new_node = Node(ALPHA, BETA, GAMMA, N, S, E, I)
    new_node.simulate()
    new_node.graph()
