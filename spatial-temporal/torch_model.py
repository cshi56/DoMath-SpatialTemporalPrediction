import torch
from torch import nn
import numpy as np
import random
import torch_geometric as pyg
from process_data import generate_dataset
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta

random.seed(1234)
torch.manual_seed(1234)


def calculate_validation_loss(model, validation_data, loss_func):
    total_loss = 0
    seq_length = len(validation_data[0][0])

    for i, (x, y) in enumerate(validation_data):
        output = None
        hidden_state = model.init_hidden()
        for graph in x:
            output, hidden_state = model(graph, hidden_state)
        loss = loss_func(output, y)
        total_loss += loss.item()
    avg_loss = total_loss / seq_length

    return avg_loss


def train_conv_rnn(model,
                   training_data,
                   epochs,
                   loss_func=nn.MSELoss,
                   optim=torch.optim.Adam,
                   lr=0.001,
                   lr_decay=1.0,
                   validation_data=None):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable parameters:', pytorch_total_params)

    hist = {'epochs': [], 'training_loss': [], 'validation_loss': []}

    loss_func = loss_func()
    learning_rate = lr
    t1 = time()
    for epoch in range(epochs):
        opt = optim(model.parameters(), lr=learning_rate)
        random.shuffle(training_data)
        total_loss = 0
        seq_length = len(training_data[0][0])

        print('Epoch ' + str(epoch + 1) + '/' + str(EPOCHS) + ': ', end='')
        total_equals = 0

        for i, (x, y) in enumerate(training_data):
            equals_to_print = int(40 * (i + 1) / len(training_data)) - total_equals
            total_equals += equals_to_print
            output = None
            hidden_state = model.init_hidden()
            for graph in x:
                output, hidden_state = model(graph, hidden_state)
            loss = loss_func(output, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            total_loss += loss.item()
            for _ in range(equals_to_print):
                print('=', end='')
        avg_loss = total_loss / seq_length
        print(' Done. ', end='')
        print('Training loss: ' + '{:.3e}'.format(avg_loss) + '. ', end='')
        if validation_data is not None:
            with torch.no_grad():
                val_loss = calculate_validation_loss(model, validation_data, loss_func)
            print('Validation loss: ' + '{:.3e}'.format(val_loss) + '. ', end='')
        print('Learning Rate: ' + '{:.3e}'.format(learning_rate) + '. ', end='')
        hist['epochs'].append(epoch)
        hist['training_loss'].append(avg_loss)
        hist['validation_loss'].append(val_loss)
        learning_rate *= lr_decay
        t2 = time()
        total_time = t2 - t1
        epochs_left = epochs - epoch - 1
        avg_time = total_time / (epoch + 1)
        time_left = epochs_left * avg_time
        print('Time left: ' + str(timedelta(seconds=time_left)))

    return hist


class ConvRNN(nn.Module):
    def __init__(self, num_nodes, num_feats, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_feats = num_feats

        self.graph_conv = pyg.nn.GCNConv(num_feats + hidden_size, hidden_size)
        self.b_matrix = nn.Parameter(torch.zeros(num_nodes, hidden_size))
        self.v_matrix = nn.Parameter(torch.zeros(hidden_size, num_feats))
        self.c_matrix = nn.Parameter(torch.zeros(num_nodes, num_feats))

        self.init_all()

    def forward(self, x, hidden_state):
        x_h_concatenated = torch.cat((x.x, hidden_state), dim=1)
        conv = self.graph_conv(x_h_concatenated, x.edge_index)
        h_pre = self.b_matrix + conv
        new_hidden = torch.sigmoid(h_pre)
        o = self.c_matrix + torch.mm(new_hidden, self.v_matrix)
        return o, new_hidden

    def init_hidden(self):
        return torch.zeros(self.num_nodes, self.hidden_size)

    def init_all(self):
        torch.nn.init.xavier_uniform_(self.v_matrix)


if __name__ == '__main__':
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    LR_DECAY = 0.99
    EPOCHS = 75
    LOSS_FUNCTION = nn.MSELoss
    OPTIMIZER = torch.optim.Adam
    TRAIN_NUM = 100
    VAL_NUM = 50
    TEST_NUM = 50
    MODEL_SAVE_PATH = 'models/conv_rnn2.pt'

    edge_index = [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                  [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]]

    dataset = generate_dataset('simple_simulation.npy', edge_index, 10, 1)
    train_data = dataset[:TRAIN_NUM]
    val_data = dataset[TRAIN_NUM: TRAIN_NUM + VAL_NUM]
    test_data = dataset[TRAIN_NUM + VAL_NUM: TRAIN_NUM + TEST_NUM + VAL_NUM]

    conv_rnn = ConvRNN(8, 4, HIDDEN_SIZE)

    history = (train_conv_rnn(conv_rnn,
                              train_data,
                              EPOCHS,
                              loss_func=LOSS_FUNCTION,
                              optim=OPTIMIZER,
                              lr=LEARNING_RATE,
                              lr_decay=LR_DECAY,
                              validation_data=val_data))

    epochs = history['epochs'][10:]
    train_loss = history['training_loss'][10:]
    val_loss = history['validation_loss'][10:]

    torch.save(conv_rnn.state_dict(), MODEL_SAVE_PATH)

    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
