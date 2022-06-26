import torch
from torch import nn
from time import time
import random
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

random.seed(42)
torch.manual_seed(42)


class AbstractRNN(nn.Module):
    def __init__(self,
                 previous_steps,
                 future_steps,
                 hidden_dimension):
        super().__init__()

        self.history = {'epochs': [], 'training_loss': [], 'validation_loss': []}
        self.previous_steps = previous_steps
        self.future_steps = future_steps
        self.hidden_dimension = hidden_dimension

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def initial_hidden(self):
        raise NotImplementedError

    def create_datasets(self, data_file, stride, train_num, val_num):
        raise NotImplementedError

    def calculate_loss(self, data, loss_func):
        losses = []

        for i, (x, y) in enumerate(data):
            output = None
            hidden_state = self.initial_hidden()
            for vector in x:
                output, hidden_state = self.forward(vector, hidden_state)
            loss = loss_func(output, y)
            losses.append(loss.item())
        avg_loss = np.asarray(losses).mean()

        return avg_loss

    def train_model(self,
                    data_file,
                    train_num,
                    val_num,
                    stride,
                    epochs,
                    loss_func=nn.MSELoss,
                    optim=torch.optim.Adam,
                    lr=0.001,
                    lr_decay=1.0,
                    batch_size=1):

        training_data, validation_data = self.create_datasets(data_file,
                                                              stride,
                                                              train_num,
                                                              val_num)

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Total number of trainable parameters:', pytorch_total_params)

        loss_func = loss_func()
        learning_rate = lr
        t1 = time()
        for epoch in range(epochs):
            opt = optim(self.parameters(), lr=learning_rate)
            random.shuffle(training_data)
            losses = []

            print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ': ', end='')
            total_equals = 0

            loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            for i, (x, y) in enumerate(training_data):
                equals_to_print = int(40 * (i + 1) / len(training_data)) - total_equals
                total_equals += equals_to_print
                output = None
                hidden_state = self.initial_hidden()

                for vector in x:
                    output, hidden_state = self.forward(vector, hidden_state)

                loss = loss + loss_func(output, y)

                if i > 0 and i % batch_size == 0:
                    loss = loss / batch_size
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)

                losses.append(loss.item())
                for _ in range(equals_to_print):
                    print('=', end='')
            with torch.no_grad():
                train_loss = self.calculate_loss(training_data, loss_func)
            print(' Done. ', end='')
            print('Training loss: ' + '{:.3e}'.format(train_loss) + '. ', end='')
            with torch.no_grad():
                val_loss = self.calculate_loss(validation_data, loss_func)
            print('Validation loss: ' + '{:.3e}'.format(val_loss) + '. ', end='')
            print('Learning Rate: ' + '{:.3e}'.format(learning_rate) + '. ', end='')
            self.history['epochs'].append(epoch + 1)
            self.history['training_loss'].append(train_loss)
            self.history['validation_loss'].append(val_loss)
            learning_rate *= lr_decay
            t2 = time()
            total_time = t2 - t1
            epochs_left = epochs - epoch - 1
            avg_time = total_time / (epoch + 1)
            time_left = epochs_left * avg_time
            print('Time left: ' + str(timedelta(seconds=time_left)))

    def predict(self, x, future_steps):
        raise NotImplementedError

    def plot_loss(self, start_epoch):
        epochs = self.history['epochs'][start_epoch - 1:]
        train_loss = self.history['training_loss'][start_epoch - 1:]
        val_loss = self.history['validation_loss'][start_epoch - 1:]
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


class AbstractLSTM(nn.Module):
    def __init__(self,
                 previous_steps,
                 future_steps,
                 hidden_dimension):
        super().__init__()

        self.history = {'epochs': [], 'training_loss': [], 'validation_loss': []}
        self.previous_steps = previous_steps
        self.future_steps = future_steps
        self.hidden_dimension = hidden_dimension

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def initial_hidden(self):
        raise NotImplementedError

    def initial_cell(self):
        raise NotImplementedError

    def create_datasets(self, data_file, stride, train_num, val_num):
        raise NotImplementedError

    def calculate_loss(self, data, loss_func):
        losses = []

        for i, (x, y) in enumerate(data):
            output = None
            hidden_state = self.initial_hidden()
            cell_state = self.initial_cell()
            for vector in x:
                output, hidden_state, cell_state = self.forward(vector, hidden_state, cell_state)
            loss = loss_func(output, y)
            losses.append(loss.item())
        avg_loss = np.asarray(losses).mean()

        return avg_loss

    def train_model(self,
                    data_file,
                    train_num,
                    val_num,
                    stride,
                    epochs,
                    loss_func=nn.MSELoss,
                    optim=torch.optim.Adam,
                    lr=0.001,
                    lr_decay=1.0,
                    batch_size=1):

        training_data, validation_data = self.create_datasets(data_file,
                                                              stride,
                                                              train_num,
                                                              val_num)

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Total number of trainable parameters:', pytorch_total_params)

        loss_func = loss_func()
        learning_rate = lr
        t1 = time()
        for epoch in range(epochs):
            opt = optim(self.parameters(), lr=learning_rate)
            random.shuffle(training_data)
            losses = []

            print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ': ', end='')
            total_equals = 0

            loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)

            for i, (x, y) in enumerate(training_data):
                equals_to_print = int(40 * (i + 1) / len(training_data)) - total_equals
                total_equals += equals_to_print
                output = None
                hidden_state = self.initial_hidden()
                cell_state = self.initial_cell()
                for vector in x:
                    output, hidden_state, cell_state = self.forward(vector, hidden_state, cell_state)

                loss = loss + loss_func(output, y)

                if i > 0 and i % batch_size == 0:
                    loss = loss / batch_size
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
                losses.append(loss.item())
                for _ in range(equals_to_print):
                    print('=', end='')
            with torch.no_grad():
                train_loss = self.calculate_loss(training_data, loss_func)
            print(' Done. ', end='')
            print('Training loss: ' + '{:.3e}'.format(train_loss) + '. ', end='')
            with torch.no_grad():
                val_loss = self.calculate_loss(validation_data, loss_func)
            print('Validation loss: ' + '{:.3e}'.format(val_loss) + '. ', end='')
            print('Learning Rate: ' + '{:.3e}'.format(learning_rate) + '. ', end='')
            self.history['epochs'].append(epoch + 1)
            self.history['training_loss'].append(train_loss)
            self.history['validation_loss'].append(val_loss)
            learning_rate *= lr_decay
            t2 = time()
            total_time = t2 - t1
            epochs_left = epochs - epoch - 1
            avg_time = total_time / (epoch + 1)
            time_left = epochs_left * avg_time
            print('Time left: ' + str(timedelta(seconds=time_left)))

    def predict(self, x, future_steps):
        raise NotImplementedError

    def plot_loss(self, start_epoch):
        epochs = self.history['epochs'][start_epoch - 1:]
        train_loss = self.history['training_loss'][start_epoch - 1:]
        val_loss = self.history['validation_loss'][start_epoch - 1:]
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

