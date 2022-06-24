import torch
from torch import nn
import random
import numpy as np
from abstract_models import AbstractRNN

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


class RNNVectorized(AbstractRNN):
    def __init__(self, num_nodes, num_feats, previous_steps, future_steps, hidden_size):
        super().__init__(previous_steps,
                         future_steps,
                         hidden_size)

        self.hidden_size = hidden_size
        self.num_feat = num_feats

        self.w_matrix = nn.Linear(hidden_size + num_feats * num_nodes, hidden_size)
        self.v_matrix = nn.Linear(hidden_size, num_feats * num_nodes * future_steps)

        self.initialize()

    def create_dataset(self, sims, stride):
        dataset = []
        simulations = sims
        simulation_length = len(simulations[0][0])
        for simulation in simulations:
            start = random.randint(0, stride - 1)
            simulation = normalize_sim(simulation)
            simulation = vectorize(simulation)
            for index in range(start, simulation_length - self.previous_steps - self.future_steps + 1, stride):
                x = torch.tensor(simulation[index: index + self.previous_steps], dtype=torch.float)
                y = torch.tensor(simulation[index + self.previous_steps: index + self.previous_steps + self.future_steps].flatten(),
                                 dtype=torch.float)
                dataset.append((x, y))
        random.shuffle(dataset)
        return dataset

    def create_datasets(self, datafile, stride, train_num, val_num):
        all_data = np.load(datafile)
        train_sims = all_data[:train_num]
        val_sims = all_data[train_num: train_num + val_num]
        train_dataset = self.create_dataset(train_sims, stride)
        val_dataset = self.create_dataset(val_sims, stride)
        return train_dataset, val_dataset

    def forward(self, x, hidden_state):
        xh_concat = torch.cat((x, hidden_state), dim=0)
        pre_hidden = self.w_matrix(xh_concat)
        new_hidden = torch.sigmoid(pre_hidden)
        output = self.v_matrix(new_hidden)
        return output, new_hidden

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.w_matrix.weight)
        torch.nn.init.xavier_uniform_(self.v_matrix.weight)
        torch.nn.init.zeros_(self.w_matrix.bias)
        torch.nn.init.zeros_(self.v_matrix.bias)

    def initial_hidden(self):
        return torch.zeros(self.hidden_size)

    def predict(self, sim, time_steps):
        pass
