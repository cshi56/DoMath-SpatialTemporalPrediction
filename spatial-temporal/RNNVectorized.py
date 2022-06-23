import torch
from torch import nn
import random
import numpy as np
from time import time
from datetime import timedelta
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


def normalize_sim(sim):
    max_pop = max([sum(node[0]) for node in sim])
    return np.asarray(sim, dtype=float) / max_pop


def create_datasets(file, previous_steps, future_steps, stride, train_num, val_num):
    all_data = np.load(file)
    train_sims = all_data[:train_num]
    val_sims = all_data[train_num: train_num + val_num]
    train_dataset = create_dataset(train_sims, previous_steps, future_steps, stride)
    val_dataset = create_dataset(val_sims, previous_steps, future_steps, stride)
    return train_dataset, val_dataset


def create_dataset(sims, previous_steps, future_steps, stride):
    dataset = []
    simulations = sims
    simulation_length = len(simulations[0][0])
    for simulation in simulations:
        start = random.randint(0, stride - 1)
        simulation = normalize_sim(simulation)
        simulation = vectorize(simulation)
        for index in range(start, simulation_length - previous_steps - future_steps + 1, stride):
            x = torch.tensor(simulation[index: index + previous_steps], dtype=torch.float)
            y = torch.tensor(simulation[index + previous_steps: index + previous_steps + future_steps].flatten(),
                             dtype=torch.float)
            dataset.append((x, y))
    random.shuffle(dataset)
    return dataset


def calculate_loss(model, data, loss_func):
    losses = []

    for i, (x, y) in enumerate(data):
        output = None
        hidden_state = model.init_hidden()
        for vector in x:
            output, hidden_state = model(vector, hidden_state)
        loss = loss_func(output, y)
        losses.append(loss.item())
    avg_loss = np.asarray(losses).mean()

    return avg_loss


def train_rnn_vectorized(hidden_size,
                         data_file,
                         train_num,
                         val_num,
                         previous_steps,
                         future_steps,
                         stride,
                         epochs,
                         loss_func=nn.MSELoss,
                         optim=torch.optim.Adam,
                         lr=0.001,
                         lr_decay=1.0):
    all_data = np.load(data_file)
    num_nodes = len(all_data[0])
    num_feats = len(all_data[0][0][0])

    training_data, validation_data = create_datasets(data_file,
                                                     previous_steps,
                                                     future_steps,
                                                     stride,
                                                     train_num,
                                                     val_num)

    model = RNNVectorized(num_nodes, num_feats, future_steps, hidden_size)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable parameters:', pytorch_total_params)

    hist = {'epochs': [], 'training_loss': [], 'validation_loss': []}

    loss_func = loss_func()
    learning_rate = lr
    t1 = time()
    for epoch in range(epochs):
        opt = optim(model.parameters(), lr=learning_rate)
        random.shuffle(training_data)
        losses = []

        print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ': ', end='')
        total_equals = 0

        for i, (x, y) in enumerate(training_data):
            equals_to_print = int(40 * (i + 1) / len(training_data)) - total_equals
            total_equals += equals_to_print
            output = None
            hidden_state = model.init_hidden()
            for vector in x:
                output, hidden_state = model(vector, hidden_state)
            loss = loss_func(output, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(loss.item())
            for _ in range(equals_to_print):
                print('=', end='')
        with torch.no_grad():
            train_loss = calculate_loss(model, training_data, loss_func)
        print(' Done. ', end='')
        print('Training loss: ' + '{:.3e}'.format(train_loss) + '. ', end='')
        with torch.no_grad():
            val_loss = calculate_loss(model, validation_data, loss_func)
        print('Validation loss: ' + '{:.3e}'.format(val_loss) + '. ', end='')
        print('Learning Rate: ' + '{:.3e}'.format(learning_rate) + '. ', end='')
        hist['epochs'].append(epoch)
        hist['training_loss'].append(train_loss)
        hist['validation_loss'].append(val_loss)
        learning_rate *= lr_decay
        t2 = time()
        total_time = t2 - t1
        epochs_left = epochs - epoch - 1
        avg_time = total_time / (epoch + 1)
        time_left = epochs_left * avg_time
        print('Time left: ' + str(timedelta(seconds=time_left)))

    return hist


class RNNVectorized(nn.Module):
    def __init__(self, num_nodes, num_feats, future_steps, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.future_steps = future_steps

        self.u_matrix = nn.Linear(num_feats * num_nodes, hidden_size, bias=False)
        self.w_matrix = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.v_matrix = nn.Linear(hidden_size, num_feats * num_nodes * future_steps, bias=False)
        self.c_bias = nn.Parameter(torch.zeros(num_feats * num_nodes * future_steps))

        self.init_all()

    def forward(self, x, hidden_state):
        pre_hidden = self.bias + self.w_matrix(hidden_state) + self.u_matrix(x)
        new_hidden = torch.sigmoid(pre_hidden)
        output = self.c_bias + self.v_matrix(new_hidden)
        return output, new_hidden

    def init_all(self):
        torch.nn.init.xavier_uniform_(self.u_matrix.weight)
        torch.nn.init.xavier_uniform_(self.w_matrix.weight)
        torch.nn.init.xavier_uniform_(self.v_matrix.weight)

    def init_hidden(self):
        return torch.zeros(self.hidden_size)


if __name__ == '__main__':
    DATA_FILE = 'data/200sims_50days_1node.npy'
    PREVIOUS_STEPS = 20
    FUTURE_STEPS = 1
    STRIDE = 5

    HIDDEN_SIZE = 64
    TRAIN_NUM = 100
    VAL_NUM = 50
    EPOCHS = 100
    LR_DECAY = 0.95

    history = train_rnn_vectorized(HIDDEN_SIZE,
                                   DATA_FILE,
                                   TRAIN_NUM,
                                   VAL_NUM,
                                   PREVIOUS_STEPS,
                                   FUTURE_STEPS,
                                   STRIDE,
                                   EPOCHS,
                                   lr_decay=LR_DECAY)

    train_loss = history['training_loss'][10:]
    val_loss = history['validation_loss'][10:]
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.show()
