import numpy as np
import matplotlib.pyplot as plt


file = 'data/fixed-parameters/150sims_50days_2nodes.npy'

data = np.load(file)

for i, sim in enumerate(data):
    for j, node in enumerate(sim):
        i_data = node[:, 2]
        plt.plot(i_data)
        if node[19][2] < 40:
            print(i, j)
            print(node[19][2])

plt.show()
