import torch
from RNNVectorized import RNNVectorized, normalize_sim, vectorize
from LSTMVectorized import LSTMVectorized
from GCRNN import GCRNN
from GCLSTM import GCLSTM
import numpy as np
import matplotlib.pyplot as plt


def unvectorize(sim, num_nodes):
    sim = np.asarray(sim)
    ret = []
    for i in range(num_nodes):
        node_data = sim[:, i:i + 4]
        ret.append(node_data)
    return np.asarray(ret)


def rnn_predict_one_step(model, input_steps):
    hidden_state = model.initial_hidden()
    output = None
    for vector in input_steps:
        output, hidden_state = model(vector, hidden_state)
    return output


def rnn_predict(model, sim, prev_steps, time_steps, num_nodes):
    model.eval()
    max_pop = max([sum(node[0]) for node in sim])
    sim = normalize_sim(sim)
    sim = vectorize(sim)
    sim = torch.tensor(sim, dtype=torch.float)
    input_steps = sim[:prev_steps]
    current_data = input_steps

    future_data = []

    for step in range(time_steps):
        next_step = rnn_predict_one_step(model, current_data)
        future_data.append(next_step.detach().numpy())
        next_step = next_step[None, :]
        current_data = torch.cat((current_data, next_step), dim=0)
        current_data = current_data[1:]

    return unvectorize(future_data, num_nodes) * max_pop


def graph_compare(model, sim, prev_steps, time_steps, num_nodes):
    real_data = sim[:time_steps]
    predicted_data = rnn_predict(model, sim, prev_steps, time_steps, num_nodes)
    for node_dex in range(num_nodes):
        plt.plot(real_data[node_dex][:prev_steps + time_steps, 2], c='black', lw=1, label='Ground truth')
        plt.plot(range(prev_steps, prev_steps + time_steps), predicted_data[node_dex][:, 2],
                 ls='dotted', lw=2, c='red', label='Predicted values')
        title = 'RNN prediction given data from days 1-' + str(prev_steps)
        plt.title(title)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Number of infected subjects')
        plt.show()


model = RNNVectorized(1, 4, 20, 1, 64)
model.load_state_dict(torch.load('models/1_nodes/vecrnn.pt'))

sims = np.load('data/fixed-parameters/150sims_50days_1nodes.npy')

for i in range(100, 150):
    sim = sims[i]
    graph_compare(model, sim, 20, 30, 1)

