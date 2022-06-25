import numpy as np
import torch
from GCRNN import GCRNN
from GCLSTM import GCLSTM
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def normalize_sim(sim):
    max_pop = max([sum(node[0]) for node in sim])
    return np.asarray(sim, dtype=float) / max_pop


def make_edge_index(num_nodes):
    first_vec = []
    second_vec = []
    for i in range(num_nodes):
        first_vec.append(i)
        first_vec.append(i)
    first_vec = first_vec[1:-1]
    for i in range(len(first_vec)):
        second_vec.append(first_vec[i] + ((-1) ** i))
    e_i = np.asarray([first_vec, second_vec])
    edge_index = torch.tensor(e_i, dtype=torch.long)
    return edge_index


def create_dataset_from_sim(sim, stride, nodes, previous_steps, future_steps):
    sim_length = len(sim[0])
    graph_data = normalize_sim(sim)
    edge_index = make_edge_index(num_nodes=nodes)

    graph_series = []

    for time in range(sim_length):
        vertices = []
        for node_index in range(len(graph_data)):
            seir = graph_data[node_index][time]
            vertices.append(seir)
        vertices = np.asarray(vertices)
        vertices = torch.tensor(vertices, dtype=torch.float)
        graph = Data(vertices, edge_index)
        graph_series.append(graph)

    # The first previous steps will be one data point, the rest will simply be the individual points
    graph_dataset = []
    datum = graph_series[0: previous_steps]
    label = graph_series[previous_steps].x
    graph_dataset.append((datum, label))

    for index in range(previous_steps, sim_length, stride):
        datum = graph_series[index]
        label = graph_series[index].x
        graph_dataset.append((datum, label))

    """"
    for index in range(0, sim_length - previous_steps - future_steps + 1, stride):
        datum = graph_series[index: index + previous_steps]
        label = graph_series[index + previous_steps].x
        graph_dataset.append((datum, label))
    """
    return graph_dataset


def create_datasets(datafile, stride, test_num, num_nodes, previous_steps, future_steps=1):
    test_dataset = []
    all_data = np.load(datafile)
    all_data = all_data[-test_num:]

    for i in range(test_num):
        mini_set = create_dataset_from_sim(all_data[i], stride, num_nodes, previous_steps, future_steps)
        test_dataset.append(mini_set)

    return test_dataset


if __name__ == '__main__':
    gcrnn_1_node = GCRNN(num_nodes=1, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gcrnn_1_node.load_state_dict(torch.load('models/1_nodes/gcrnn_20prev_1fut.pt'))
    gcrnn_2_node = GCRNN(num_nodes=2, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gcrnn_2_node.load_state_dict(torch.load('models/2_nodes/gcrnn_20prev_1fut.pt'))
    gcrnn_10_node = GCRNN(num_nodes=10, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gcrnn_10_node.load_state_dict(torch.load('models/10_nodes/gcrnn_20prev_1fut.pt'))

    gclstm_1_node = GCLSTM(num_nodes=1, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gclstm_1_node.load_state_dict(torch.load('models/1_nodes/gclstm_20prev_1fut.pt'))
    gclstm_2_node = GCLSTM(num_nodes=2, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gclstm_2_node.load_state_dict(torch.load('models/2_nodes/gclstm_20prev_1fut.pt'))
    gclstm_10_node = GCLSTM(num_nodes=10, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gclstm_10_node.load_state_dict(torch.load('models/10_nodes/gclstm_20prev_1fut.pt'))

    datapath_list = ['data/200sims_50days_1nodes.npy',
                     'data/200sims_50days_2nodes.npy',
                     'data/200sims_50days_10nodes.npy']

    test_num = 50
    previous_steps = 20

    test_dataset_1node = create_datasets('data/200sims_50days_1nodes.npy', 1, test_num, num_nodes=1, previous_steps=previous_steps)
    test_dataset_2node = create_datasets('data/200sims_50days_2nodes.npy', 1, test_num, num_nodes=2, previous_steps=previous_steps)
    test_dataset_10node = create_datasets('data/200sims_50days_10nodes.npy', 1, test_num, num_nodes=10,
                                          previous_steps=previous_steps)

    edge_index_1 = make_edge_index(1)

    # Testing for 1 node, currently plots one simulation
    actual_1 = [[] for i in range(test_num)]
    predicted_1 = [[] for i in range(test_num)]

    actual_2 = [[] for i in range(test_num)]
    predicted_2 = [[] for i in range(test_num)]
    # fig, axis = plt.subplots(len(test_dataset_1node), 1)

    for sim in range(len(test_dataset_1node)):
        for i, (x, y) in enumerate(test_dataset_1node[sim]):
            if i == 0:
                h_n = torch.zeros(1, 64)
                for j in range(len(x)):
                    predicted_1[sim].append(x[j].x[0].detach().numpy())
                    actual_1[sim].append(x[j].x[0].detach().numpy())
                    out, h_n = gcrnn_1_node(x[j], h_n)
            else:
                out, h_n = gcrnn_1_node(x, h_n)
            predicted_1[sim].append(out[0].detach().numpy())
            actual_1[sim].append(y[0].detach().numpy())

    plt.plot(range(len(predicted_1[4])), predicted_1[4], ls="dotted")
    plt.plot(range(len(actual_1[4])), actual_1[4])
    plt.show()

    for sim in range(len(test_dataset_2node)):
        for i, (x, y) in enumerate(test_dataset_2node[sim]):
            if i == 0:
                h_n = torch.zeros(2, 64)
                for j in range(len(x)):
                    predicted_2[sim].append(x[j].x.detach().numpy())
                    actual_2[sim].append(x[j].x.detach().numpy())
                    out, h_n = gcrnn_2_node(x[j], h_n)
            else:
                out, h_n = gcrnn_2_node(x, h_n)

            predicted_2[sim].append(out.detach().numpy())
            actual_2[sim].append(y.detach().numpy())
            #print("out: " + out)

    predicted_2 = np.asarray(predicted_2)
    temp_node_p = np.ndarray.tolist(predicted_2[6, :, 1, :])
    actual_2 = np.asarray(actual_2)
    temp_node_a = np.ndarray.tolist(actual_2[6, :, 1, :])
    plt.plot(range(len(temp_node_p)), temp_node_p, ls="dotted")
    plt.plot(range(len(temp_node_a)), temp_node_a)
    plt.show()