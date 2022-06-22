import torch
from load_data import preprocess_as_temporal
import numpy as np
import matplotlib.pyplot as plt
from load_data import simulate

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layers=64):
        super(RNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.rnn1 = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_layers, batch_first=True)
        self.rnn2 = torch.nn.RNN(input_size=self.hidden_layers, hidden_size=self.hidden_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_layers, 40)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output, h = self.rnn1(x)
        output, h = self.rnn2(output, h)
        output = self.linear(output)
        output = self.relu(output)
        return output


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.lstm1 = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_layers, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=self.hidden_layers, hidden_size=self.hidden_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_layers, 40)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output, (h, c) = self.lstm1(x)
        output, (h, c) = self.lstm2(output, (h, c))
        output = self.linear(output)
        output = self.relu(output)
        return output


def training_loop(n_epochs, optimizer, model, loss_fn, dataloader):
    for epoch in range(0, n_epochs):
        for batch, (x, y) in enumerate(dataloader):

            x = x.float()
            x = torch.flatten(x, 1)

            output_train = model(x)  # forwards pass

            y = torch.flatten(y, 1)
            y = y.float()
            loss_train = loss_fn(output_train, y)  # calculate loss

            optimizer.zero_grad()  # set gradients to zero
            loss_train.backward()  # backwards pass
            optimizer.step()  # update model parameters

            if epoch == 1 or epoch % 5 == 0:
                print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")


def test_loop(model, loss, dataloader):

    actual_data = []
    predicted_data = []
    for time, (x, y) in enumerate(dataloader):
        x = x.float()

        x = torch.flatten(x, 1)
        y_hat = model(x)

        y_hat = torch.reshape(y_hat, (10, 4))
        print(y_hat)
        predict = y_hat[:, 2]
        actual = y[:, :, 2]

        actual_data.append(torch.sum(actual).detach().numpy())
        predicted_data.append(torch.sum(predict).detach().numpy())


    # Plots total infected data
    plt.plot(range(len(actual_data)), actual_data, label=['actual I'])
    plt.plot(range(len(predicted_data)), predicted_data,
             label=['predicted I'], ls='dotted', lw=2.5)
    plt.legend()
    plt.show()

def plot_one_node(dataloader):
    node = []

    for time, (x, y) in enumerate(dataloader):
        node.append(y[0, 0, :].detach().numpy())
    plt.plot(range(len(node)), node)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_name = 'data.npz'
    NODES = 10
    STEPS = 500
    PREVIOUS_STEPS = 20
    future_steps = 1
    stride = 1

    #sim = simulate(STEPS, NODES, file_name)

    NUM_EPOCHS = 50

    train_loader, test_loader = preprocess_as_temporal(file_name, PREVIOUS_STEPS, future_steps=future_steps, stride=stride)

    rnn = RNN(input_size=840)
    lstm = LSTM(input_size=840)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()

    training_loop(NUM_EPOCHS, optimizer, rnn, loss, train_loader)
    training_loop(NUM_EPOCHS, optimizer, lstm, loss, train_loader)
    test_loop(rnn, loss, test_loader)
    test_loop(lstm, loss, test_loader)

