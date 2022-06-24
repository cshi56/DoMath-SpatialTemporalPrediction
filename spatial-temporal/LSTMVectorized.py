import torch
from torch import nn
import random
import numpy as np
from abstract_models import AbstractLSTM

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


class LSTMVectorized(AbstractLSTM):
    def __init__(self, num_nodes, num_feats, previous_steps, future_steps, hidden_size):
        super().__init__(previous_steps,
                         future_steps,
                         hidden_size)

        self.hidden_size = hidden_size
        self.num_feat = num_feats

        self.wf_matrix = nn.Linear(hidden_size + num_feats * num_nodes, hidden_size)
        self.wi_matrix = nn.Linear(hidden_size + num_feats * num_nodes, hidden_size)
        self.wo_matrix = nn.Linear(hidden_size + num_feats * num_nodes, hidden_size)

        self.wc_matrix = nn.Linear(hidden_size + num_feats * num_nodes, hidden_size)

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

    def forward(self, x, hidden_state, cell_state):
        xh_concat = torch.cat((x, hidden_state), dim=0)
        ft = torch.sigmoid(self.wf_matrix(xh_concat))
        it = torch.sigmoid(self.wi_matrix(xh_concat))
        ot = torch.sigmoid(self.wo_matrix(xh_concat))

        cell_prime = torch.tanh(self.wc_matrix(xh_concat))
        new_cell = (ft * cell_state) + (it * cell_prime)
        new_hidden = ot * torch.tanh(new_cell)

        real_output = self.v_matrix(ot)

        return real_output, new_hidden, new_cell

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.wf_matrix.weight)
        torch.nn.init.xavier_uniform_(self.wi_matrix.weight)
        torch.nn.init.xavier_uniform_(self.wo_matrix.weight)
        torch.nn.init.xavier_uniform_(self.wc_matrix.weight)
        torch.nn.init.xavier_uniform_(self.v_matrix.weight)

        torch.nn.init.zeros_(self.wf_matrix.bias)
        torch.nn.init.zeros_(self.wi_matrix.bias)
        torch.nn.init.zeros_(self.wo_matrix.bias)
        torch.nn.init.zeros_(self.wc_matrix.bias)
        torch.nn.init.zeros_(self.v_matrix.bias)

    def initial_hidden(self):
        return torch.zeros(self.hidden_size)

    def initial_cell(self):
        return torch.zeros(self.hidden_size)

    def predict(self, sim, time_steps):
        future_data = []
        current_input = x
        print(x)
        for step in range(time_steps):
            output = None
            hidden_state = self.initial_hidden()
            for vector in current_input:
                output, hidden_state = self.forward(vector, hidden_state)
            future_data.append(output)
            current_input = current_input[1:]
            current_input = None
