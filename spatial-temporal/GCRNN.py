import torch
from torch import nn
import random
import numpy as np
from abstract_models import AbstractRNN
import torch_geometric as pyg
from torch_geometric.data import Data
import matplotlib.pyplot as plt

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def vectorize(sim):
    vec = sim[0]
    for node_data in sim[1:]:
        vec = np.append(vec, node_data, axis=1)
    return vec


def un_vectorize(sim, num_feats):
    pass


def normalize_sim(sim):
    max_pop = max([sum(node[0]) for node in sim])
    return np.asarray(sim, dtype=float) / max_pop


class GCRNN(AbstractRNN):
    def __init__(self, num_nodes, num_feats, previous_steps, future_steps, hidden_size):
        super().__init__(previous_steps,
                         future_steps,
                         hidden_size)

        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.edge_index = None
        self.make_edge_index()

        self.graph_conv = pyg.nn.GCNConv(num_feats + hidden_size, hidden_size)
        self.b_matrix = nn.Parameter(torch.zeros(num_nodes, hidden_size))
        self.v_matrix = nn.Parameter(torch.zeros(hidden_size, num_feats))
        self.c_matrix = nn.Parameter(torch.zeros(num_nodes, num_feats))

        self.initialize()

    def make_edge_index(self):
        first_vec = []
        second_vec = []
        for i in range(self.num_nodes):
            first_vec.append(i)
            first_vec.append(i)
        first_vec = first_vec[1:-1]
        for i in range(len(first_vec)):
            second_vec.append(first_vec[i] + ((-1) ** i))
        e_i = np.asarray([first_vec, second_vec])
        self.edge_index = torch.tensor(e_i, dtype=torch.long)

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

    def create_datasets(self, datafile, stride, train_num, val_num):

        train_dataset = []
        val_dataset = []
        all_data = np.load(datafile)

        for i in range(train_num):
            mini_set = self.create_dataset_from_sim(all_data[i], stride)
            train_dataset.extend(mini_set)

        for i in range(train_num, train_num + val_num):
            mini_set = self.create_dataset_from_sim(all_data[i], stride)
            val_dataset.extend(mini_set)

        return train_dataset, val_dataset

    def forward(self, x, hidden_state):
        x_h_concatenated = torch.cat((x.x, hidden_state), dim=1)
        conv = self.graph_conv(x_h_concatenated, x.edge_index)
        h_pre = self.b_matrix + conv
        new_hidden = torch.sigmoid(h_pre)
        o = self.c_matrix + torch.mm(new_hidden, self.v_matrix)
        return o, new_hidden

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.v_matrix)

    def initial_hidden(self):
        return torch.zeros(self.num_nodes, self.hidden_size)

    def predict(self, sim, time_steps):
        pass
