import numpy as np
import torch
from GCRNN import GCRNN
from GCLSTM import GCLSTM


def create_dataset_from_sim(self, sim, stride):
    sim_length = len(sim[0])
    graph_data = normalize_sim(sim)

    graph_series = []

    for time in range(sim_length):
        vertices = []
        for node_index in range(len(graph_data)):
            seir = graph_data[node_index][time]
            vertices.append(seir)
        vertices = np.asarray(vertices)
        vertices = torch.tensor(vertices, dtype=torch.float)
        graph = Data(vertices, self.edge_index)
        graph_series.append(graph)

    graph_dataset = []
    start = random.randint(0, stride - 1)

    for index in range(start, sim_length - self.previous_steps - self.future_steps + 1, stride):
        datum = graph_series[index: index + self.previous_steps]
        label = graph_series[index + self.previous_steps].x
        graph_dataset.append((datum, label))

    return graph_dataset


def create_datasets(data, stride):
    train_dataset = []
    all_data = np.load(datafile)

    for i in range(train_num):
        mini_set = self.create_dataset_from_sim(all_data[i], stride)
        train_dataset.extend(mini_set)


    return train_dataset


if __name__ == '__main__':
    gcrnn_1_node = GCRNN(num_nodes=1, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gcrnn_1_node.load_state_dict(torch.load('models/1_nodes/gcrnn_20prev_1fut.pt'))
    gcrnn_2_node = GCRNN(num_nodes=2, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gcrnn_2_node.load_state_dict(torch.load('models/2_nodes/gcrnn_20prev_1fut.pt'))
    gclstm_1_node = GCLSTM(num_nodes=1, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gclstm_1_node.load_state_dict(torch.load('models/1_nodes/gclstm_20prev_1fut.pt'))
    gclstm_2_node = GCLSTM(num_nodes=2, num_feats=4, previous_steps=20, future_steps=1, hidden_size=64)
    gclstm_2_node.load_state_dict(torch.load('models/2_nodes/gclstm_20prev_1fut.pt'))

    datapath_list = ['data/200sims_50days_1nodes.npy',
                         'data/200sims_50days_2nodes.npy',
                         'data/200sims_50days_10nodes.npy']

    data1 = np.load('data/200sims_50days_1nodes.npy')
    data2 = np.load('data/200sims_50days_2nodes.npy')

    data1 = data1[150:]
    data2 = data2[150:]
