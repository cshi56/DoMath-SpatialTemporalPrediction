
import numpy as np
import matplotlib.pyplot as plt
import torch
from RNNVectorized import RNNVectorized, normalize_sim, vectorize
from LSTMVectorized import LSTMVectorized


def rnn_predict_one_step(model, input_steps):
    hidden_state = model.initial_hidden()
    for vector in input_steps:
        output, hidden_state = model(vector, hidden_state)
    return output

def rnn_predict(model, sim, prev_steps, time_steps, num_nodes):
    sim = normalize_sim(sim)
    sim = vectorize(sim)
    sim = torch.tensor(sim, dtype=torch.float)
    input_steps = sim[:prev_steps]
    output = rnn_predict_one_step(model, input_steps)
    future_steps = torch.zeros([prev_steps + time_steps, 4 * num_nodes])
    future_steps[:prev_steps, :] = input_steps
    future_steps[prev_steps, :] = output

    for step in range((time_steps - 1)):
        input_steps = future_steps[(step+1):(step+1+prev_steps), :]
        output = rnn_predict_one_step(model, input_steps)
        future_steps[prev_steps+step+1, :] = output

    pred = future_steps[prev_steps:]
    return pred.detach().numpy(), sim[prev_steps:].numpy()

def lstm_predict_one_step(model, input_steps):
    hidden_state = model.initial_hidden()
    cell_state = model.initial_cell()
    for vector in input_steps:
        output, hidden_state, cell_state = model(vector, hidden_state, cell_state)
    return output

def lstm_predict(model, sim, prev_steps, time_steps, num_nodes):
    sim = normalize_sim(sim)
    sim = vectorize(sim)
    sim = torch.tensor(sim, dtype=torch.float)
    input_steps = sim[:prev_steps]
    output = lstm_predict_one_step(model, input_steps)
    future_steps = torch.zeros([prev_steps + time_steps, 4 * num_nodes])
    future_steps[:prev_steps, :] = input_steps
    future_steps[prev_steps, :] = output

    for step in range((time_steps - 1)):
        input_steps = future_steps[(step+1):(step+1+prev_steps), :]
        print("lstm input steps")
        print(input_steps)
        output = lstm_predict_one_step(model, input_steps)
        print("lstm output")
        print(output)
        future_steps[prev_steps+step+1, :] = output
    pred = future_steps[prev_steps:]
    return pred.detach().numpy(), sim[prev_steps:].numpy()

def average_testing_loss(predictions, true_labels, nodes):

    predictions = predictions * 50000
    true_labels = true_labels * 50000
    percentage_error = 0
    print(predictions)
    print(true_labels)

    for idx in range(predictions.shape[0]):
        for node in range(nodes):
            print(node)
            percentage_error += (np.abs(predictions[idx, node * 4 + 1] - true_labels[idx, node * 4 + 1]) / true_labels[idx, node * 4 + 1]) * 100

    return percentage_error / predictions.shape[0]




if __name__ == '__main__':

    PREVIOUS_STEPS = 20
    FUTURE_STEPS = 1
    TIME_STEPS = 30
    HIDDEN_SIZE = 64
    RNN_LOSS = []
    LSTM_LOSS = []

    nodes_list = [2]
    datapath_list = ['../data/200sims_50days_1nodes.npy',
                     '../data/200sims_50days_2nodes.npy',
                     '../data/200sims_50days_10nodes.npy',
                     '../data/200sims_50days_20nodes.npy']

    for dex, nodes in enumerate(nodes_list):
        datapath = '../data/200sims_50days_2nodes.npy' #datapath_list[dex]
        simulations = np.load(datapath)
        simulations = simulations[155:156]

        rnn_path = '../models/' + str(nodes) + '_nodes/rnn_vectorized_20prev_1fut.pt'
        lstm_path = '../models/' + str(nodes) + '_nodes/lstm_vectorized_20prev_1fut.pt'

        rnn_model = RNNVectorized(nodes, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
        rnn_model.load_state_dict(torch.load(rnn_path))
        rnn_model.eval()

        lstm_model = LSTMVectorized(nodes, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
        lstm_model.load_state_dict(torch.load(lstm_path))
        lstm_model.eval()

        for sim in simulations:
            rnn_pred, rnn_true = rnn_predict(rnn_model, sim, PREVIOUS_STEPS, TIME_STEPS, nodes)
            lstm_pred, lstm_true = lstm_predict(lstm_model, sim, PREVIOUS_STEPS, TIME_STEPS, nodes)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(rnn_pred[:, 1], label="rnn pred")
            ax1.plot(rnn_true[:, 1], label="rnn true")
            ax2.plot(lstm_pred[:, 1], label="lstm pred")
            ax2.plot(lstm_true[:, 1], label="lstm true")
            ax1.legend()
            ax2.legend()
            plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(rnn_pred[:, 5], label="rnn pred")
            ax1.plot(rnn_true[:, 5], label="rnn true")
            ax2.plot(lstm_pred[:, 5], label="lstm pred")
            ax2.plot(lstm_true[:, 5], label="lstm true")
            ax1.legend()
            ax2.legend()
            plt.show()


            RNN_LOSS.append(average_testing_loss(rnn_pred, rnn_true, nodes))
            LSTM_LOSS.append(average_testing_loss(lstm_pred, lstm_true, nodes))

    print(RNN_LOSS)
    print(LSTM_LOSS)





"""

model = RNNVectorized(NODES, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
model.load_state_dict(torch.load('../models/2_nodes/rnn_vectorized_20prev_1fut.pt'))
model.eval()

print(model)
print(model(v[0, :], model.initial_hidden()))

pred, true = predict(model, a[151], 20, 30, 2)
print(pred)
print(true)
plt.plot(pred[:, 1] * 50000, label='predicted')
plt.plot(true[:, 1] * 50000, label='ground truth')
#plt.plot(a[170][1, 20:, 1], label="unnormalized")
plt.legend()
plt.ylabel('Infected Population')
plt.xlabel('Day')
plt.show()




a = np.load('../data/200sims_50days_2nodes.npy')
print(a.shape)

test_simulation = a[152]
print(test_simulation.shape)
test_simulation = normalize_sim(test_simulation)
v = vectorize(test_simulation)
v = torch.tensor(v, dtype=torch.float)
print(v.shape)
print(v)

NODES = 2
"""