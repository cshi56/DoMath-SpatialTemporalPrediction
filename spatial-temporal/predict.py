import torch
from RNNVectorized import RNNVectorized, normalize_sim, vectorize
from LSTMVectorized import LSTMVectorized
from GCRNN import GCRNN
from GCLSTM import GCLSTM
import numpy as np
import matplotlib.pyplot as plt
from simulation import Node, Simulation


def unvectorize(sim, num_nodes):
    sim = np.asarray(sim)
    ret = []
    for i in range(num_nodes):
        node_data = sim[:, i * 4:i * 4 + 4]
        ret.append(node_data)
    return np.asarray(ret)


def rnn_predict_one_step(model, input_steps):
    hidden_state = model.initial_hidden()
    output = None
    for vector in input_steps:
        output, hidden_state = model(vector, hidden_state)
    return output


def lstm_predict_one_step(model, input_steps):
    hidden_state = model.initial_hidden()
    cell_state = model.initial_hidden()
    output = None
    for vector in input_steps:
        output, hidden_state, cell_state = model(vector, hidden_state, cell_state)
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


def lstm_predict(model, sim, prev_steps, time_steps, num_nodes):
    model.eval()
    max_pop = max([sum(node[0]) for node in sim])
    sim = normalize_sim(sim)
    sim = vectorize(sim)
    sim = torch.tensor(sim, dtype=torch.float)
    input_steps = sim[:prev_steps]
    current_data = input_steps

    future_data = []

    for step in range(time_steps):
        next_step = lstm_predict_one_step(model, current_data)
        future_data.append(next_step.detach().numpy())
        next_step = next_step[None, :]
        current_data = torch.cat((current_data, next_step), dim=0)
        current_data = current_data[1:]

    return unvectorize(future_data, num_nodes) * max_pop


def graph_compare_rnn(model, sim, prev_steps, time_steps, num_nodes, num_hidden):
    real_data = sim[:, :time_steps + prev_steps]
    predicted_data = rnn_predict(model, sim, prev_steps, time_steps, num_nodes)
    for node_dex in range(num_nodes):
        plt.plot(real_data[node_dex][:prev_steps + time_steps, 2], c='black', lw=1, label='Ground truth')
        plt.plot(range(prev_steps, prev_steps + time_steps), predicted_data[node_dex][:, 2],
                 ls='dotted', lw=2, c='red', label='Predicted values')
        title = 'RNN prediction given data from days 1-' + str(prev_steps) + "\nhidden features: " + str(num_hidden)
        plt.title(title)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Number of infected subjects')
        plt.show()


def graph_compare_lstm(model, sim, prev_steps, time_steps, num_nodes, num_hidden):
    real_data = sim[:, :time_steps + prev_steps]
    predicted_data = lstm_predict(model, sim, prev_steps, time_steps, num_nodes)
    for node_dex in range(num_nodes):
        plt.plot(real_data[node_dex][:prev_steps + time_steps, 2], c='black', lw=1, label='Ground truth')
        plt.plot(range(prev_steps, prev_steps + time_steps), predicted_data[node_dex][:, 2],
                 ls='dotted', lw=2, c='red', label='Predicted values')
        title = 'RNN prediction given data from days 1-' + str(prev_steps) + "\nhidden features: " + str(num_hidden)
        plt.title(title)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Number of infected subjects')
        plt.show()


def graph_compare_rnn_lstm(rnn_model, lstm_model, sim, prev_steps, time_steps, num_nodes, num_sims, num_hidden):
    if num_nodes == 1:
        rows, cols = 1, 1
    elif num_nodes == 2:
        rows, cols = 1, 2
    elif num_nodes == 10:
        rows, cols = 2, 5
    elif num_nodes == 20:
        rows, cols = 4, 5
    else:
        rows, cols = 1, num_nodes

    real_data = sim[:, :time_steps + prev_steps]
    predicted_data_rnn = rnn_predict(rnn_model, sim, prev_steps, time_steps, num_nodes)
    predicted_data_lstm = lstm_predict(lstm_model, sim, prev_steps, time_steps, num_nodes)

    range_sims = []

    for simnum in range(num_sims):
        sim = Simulation()
        for node in range(num_nodes):
            last_step = real_data[node][prev_steps - 1]
            s = last_step[0]
            e = last_step[1]
            i = last_step[2]
            r = last_step[3]
            n = s + e + i + r
            node = Node(0.1, 0.4, 0.05, n, s, e, i)
            sim.add_node(node)

        for _ in range(time_steps):
            sim.simulate_single_time_unit()

        fut = []
        for node in range(num_nodes):
            future = np.asarray(sim.nodes[node].unit_time_history)
            fut.append(future)

        range_sims.append(fut)

    range_sims = np.asarray(range_sims)

    fig = plt.figure(1, figsize=(10, 8))

    for node_dex in range(num_nodes):
        ax = fig.add_subplot(rows, cols, node_dex + 1)
        for simul in range_sims:
            ax.plot(range(prev_steps - 1, prev_steps + time_steps), simul[node_dex][:, 3],  c='yellow', alpha=0.1)
        if node_dex == 0:
            ax.plot(0, lw=1, c='black', label='Ground truth')
            ax.plot(0, ls='dotted', lw=2, c='red', label='RNN predicted values')
            ax.plot(0, ls='dotted', lw=2, c='green', label='LSTM predicted values')
            ax.plot(0, c='yellow', label='Range')
        ax.plot(real_data[node_dex][:prev_steps + time_steps, 2], c='black', lw=1)
        ax.plot(range(prev_steps, prev_steps + time_steps), predicted_data_rnn[node_dex][:, 2],
                 ls='dotted', lw=2, c='red')
        ax.plot(range(prev_steps, prev_steps + time_steps), predicted_data_lstm[node_dex][:, 2],
                 ls='dotted', lw=2, c='green')
        ax.set_title('Node ' + str(node_dex + 1))
        ax.set(xlabel='Days', ylabel='Infected subjects')

    title = 'RNN and LSTM predictions given data from days 1-' + str(prev_steps) + "\nhidden features: " + str(num_hidden)
    plt.suptitle(title)
    fig.legend()
    plt.show()


if __name__ == '__main__':
    hidden_feats = 128
    num_nodes = 1
    rnn = RNNVectorized(num_nodes, 4, 20, 1, 64)
    rnn.load_state_dict(torch.load('models/1_nodes/vecrnn.pt'))

    lstm = LSTMVectorized(num_nodes, 4, 20, 1, hidden_feats)
    lstm.load_state_dict(torch.load('models/1_nodes/veclstm128.pt'))

    sims = np.load('data/fixed-parameters/150sims_50days_1nodes.npy')

    for i in range(100, 150):
        sim = sims[i]
        graph_compare_lstm(lstm, sim, 20, 30, num_nodes, hidden_feats)

