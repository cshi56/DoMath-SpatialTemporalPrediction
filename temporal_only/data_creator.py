from simulation import Simulation
import numpy as np
import random

with open('data_200_sims.npz', 'wb') as f:
    all_data = []
    for _ in range(200):
        beta = random.uniform(.1, .5)
        a = random.uniform(.07, .14)
        gamma = random.uniform(.02, .07)
        n = 500000
        i = random.randrange(5, 1000)
        s = n - i

        sim = Simulation(beta, a, gamma, n, s=s, i=i)
        sim.simulate_till_end()
        all_data.append(sim.unit_time_data)

        np.savez(f, *all_data)
        print('Done with simulation ' + str(_))
