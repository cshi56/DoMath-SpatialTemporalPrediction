import random

import numpy as np
from simulation import Node, Simulation
import matplotlib.pyplot as plt

np.random.seed(42)


def create_random_node(alpha, gamma, n, i):
    s = n - i
    e = 0
    beta = np.random.uniform(0.1, 0.5)
    ret = Node(alpha, beta, gamma, n, s, e, i)
    return ret


if __name__ == '__main__':
    NUM_SIMS = 200
    N = 50000
    I_INIT = 5
    E_INIT = 0
    R_INIT = 0
    S_INIT = N - I_INIT - E_INIT - R_INIT
    TIME_STEPS = 50
    NODES = 20
    FILE_PATH = 'data/200sims_50days_20nodes'

    all_data = []

    for _ in range(NUM_SIMS):
        print(_)
        sim = Simulation()
        alpha = np.random.uniform(0.07, 0.14)
        gamma = np.random.uniform(0.02, 0.07)

        for _ in range(NODES):
            beta = np.random.uniform(0.1, 0.5)
            node = Node(alpha, beta, gamma, N, S_INIT, E_INIT, I_INIT)
            sim.add_node(node)

        sim.populate_diffusion_matrix()

        for time in range(TIME_STEPS):
            sim.simulate_single_time_unit()
        nodes_data = []
        for node in sim.nodes:
            node_data = np.asarray(node.unit_time_history)[1:]
            node_data = node_data[:, 1:]
            nodes_data.append(node_data)
        all_data.append(nodes_data)

    all_data = np.asarray(all_data)
    np.save(FILE_PATH, all_data)
    for data in all_data[:2]:
        for node in data:
            i = node[:, 2]
            plt.plot(i)
            plt.show()