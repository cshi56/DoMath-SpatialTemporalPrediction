from simulation import Simulation
from simulation import Node
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

torch.random.seed()


def simulate(steps, nodes, file):
    new_sim = Simulation()
    for _ in range(nodes):
        alpha = 0.1
        beta = random.uniform(0.1, 0.5)
        gamma = 0.01
        n = random.randrange(10000, 100000)
        if _ == 0:
            i = 10
        else:
            i = 0
        s = n - i
        e = 0
        node_to_add = Node(alpha, beta, gamma, n, s, 0, i)
        new_sim.add_node(node_to_add)

    new_sim.populate_diffusion_matrix()
    print(new_sim.diffusion_matrix)

    for _ in range(steps):
        new_sim.simulate_single_time_unit()
        print("simulating time step" + str(_))

    save = []
    for node in new_sim.nodes:
        save.append(node.unit_time_history)

    np.savez(file, save)

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

    return new_sim


def make_data(data, prior_steps, future_steps=1, stride=1):
    holder = []
    print(data[0].shape)
    pop = 0
    for n in range(data[0].shape[0]):
        pop += np.sum(data[0][n, 0, :])
    print(pop)
    for d in data:
        print(d.shape)
        for i in range(d.shape[1]):
            holder.append(d[:, i, 1:])
    holder = np.asarray(holder)

    my_x = []
    my_y = []

    for i in range(0, d.shape[1] - prior_steps - future_steps, stride):
        x_datum = holder[i: i + prior_steps + future_steps, :, :]
        y_datum = holder[i + prior_steps + future_steps, :, :]
        my_x.append(x_datum)
        my_y.append(y_datum)

    my_x = np.asarray(my_x)
    my_y = np.asarray(my_y)

    my_x = my_x / pop
    my_y = my_y / pop
    return my_x, my_y

def preprocess_as_temporal(file, prior_steps, future_steps=1, stride=1):
    all_data = []
    temp = np.load(file)
    file_names = temp.files
    for name in file_names:
        all_data.append(temp[name])

    my_x, my_y = make_data(all_data, prior_steps, future_steps=future_steps, stride=stride)
    """"
    t = []
    for arr in my_y:
        t.append(arr[0, :])


    plt.plot(range(len(t)), t)
    plt.show()
    """
    #print(my_x.shape)
    #print(my_y.shape)
    tensor_x = torch.from_numpy(my_x)  # transform to torch tensor
    tensor_y = torch.from_numpy(my_y)


    train_x = tensor_x[:241]
    test_x = tensor_x[241:]

    train_y = tensor_y[:241]
    test_y = tensor_y[241:]

    train_set = TensorDataset(train_x, train_y)  # create your dataset
    test_set = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_set, batch_size=20, shuffle=True)  # create your dataloader
    test_loader = DataLoader(test_set)

    return train_loader, test_loader


if __name__ == '__main__':
    file_name = 'data.npz'
    NODES = 10
    STEPS = 500
    PREVIOUS_STEPS = 20
    future_steps = 1
    stride = 1
    #sim = simulate(STEPS, NODES, file_name)

    #sim.nodes[0].graph()
    """
    time_data = data[0, 0, :, 0] / NODES
    s_data = data[0, 0, :, 1]
    e_data = data[0, 0, :, 2]
    i_data = data[0, 0, :, 3]
    r_data = data[0, 0, :, 4]

    plt.plot(time_data, e_data, label='Exposed subjects')
    plt.plot(time_data, i_data, label='Infected subjects')
    plt.legend()
    plt.show()
"""
    train_loader, test_loader = preprocess_as_temporal(file_name, PREVIOUS_STEPS, future_steps=future_steps, stride=stride)

    for batch, (x, y) in enumerate(train_loader):
        t = x
