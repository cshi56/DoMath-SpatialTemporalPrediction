import matplotlib.pyplot as plt

from simulation import Node, Simulation
import numpy as np

np.random.seed(42)


def generate_100(alpha, beta, gamma, n, s, e, i, num_same_first_days, total_time, num_sims):
    sim = Simulation()
    node = Node(alpha, beta, gamma, n, s, e, i)
    sim.add_node(node)
    sim.diffusion_matrix = [[0]]
    for _ in range(num_same_first_days):
        sim.simulate_single_time_unit()
    same_first_days = np.asarray(sim.nodes[0].unit_time_history[:num_same_first_days])
    initial_day_for_rest = same_first_days[-1]
    s_init = initial_day_for_rest[1]
    e_init = initial_day_for_rest[2]
    i_init = initial_day_for_rest[3]
    var_sims = []
    for sim in range(num_sims):
        print('sim ' + str(sim + 1))
        var_sim = Simulation()
        var_node = Node(alpha, beta, gamma, n, s_init, e_init, i_init)
        var_sim.add_node(var_node)
        for time in range(total_time - num_same_first_days):
            var_sim.simulate_single_time_unit()
        new_steps = var_sim.nodes[0].unit_time_history[1:]
        total_var_sim = np.concatenate((same_first_days, new_steps), axis=0)
        var_sims.append(total_var_sim)

    var_sims = np.asarray(var_sims)
    for sim in var_sims:
        this_i = sim[:, 3]
        plt.plot(range(1, total_time + 1), this_i, c='blue', alpha=min(1, 10000 / num_sims))
    title = 'α =  ' + '{:.3f}'.format(alpha) + ', '
    title += 'β =  ' + '{:.3f}'.format(beta) + ', '
    title += 'γ =  ' + '{:.3f}'.format(gamma) + ', '
    title += str(num_same_first_days) + ' same first time steps'
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Number of infected subjects')
    plt.show()
    return np.asarray(var_sims)


if __name__ == '__main__':
    ALPHA = 0.1
    BETA = 0.4
    GAMMA = 0.05
    N = 500000
    I = 10
    E = 0
    S = N - I - E
    INITIAL_DAYS = 1
    TOTAL_TIME = 20
    NUM_SIMS = 1000

    generate_100(ALPHA, BETA, GAMMA, N, S, E, I, INITIAL_DAYS, TOTAL_TIME, NUM_SIMS)
