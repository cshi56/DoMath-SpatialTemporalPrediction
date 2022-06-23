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
    N = 500000
    I_INIT = 5
    E_INIT = 0
    R_INIT = 0
    S_INIT = N - I_INIT - E_INIT - R_INIT
    TIME_STEPS = 50
    FILE_PATH = 'data/200sims_50days_2nodes'

    all_data = []

    for _ in range(NUM_SIMS):
        alpha = np.random.uniform(0.07, 0.14)
        beta = np.random.uniform(0.1, 0.5)
        gamma = np.random.uniform(0.02, 0.07)
        alpha2 = np.random.uniform(0.07, 0.14)
        beta2 = np.random.uniform(0.1, 0.5)
        gamma2 = np.random.uniform(0.02, 0.07)
        sim = Simulation()
        node1 = Node(alpha, beta, gamma, N, S_INIT, E_INIT, I_INIT)
        node2 = Node(alpha2, beta2, gamma2, 500000, 500000, 0, 0)
        sim.add_node(node1)
        sim.add_node(node2)
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
