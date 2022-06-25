import torch
from load_data import preprocess_as_temporal
import numpy as np
import matplotlib.pyplot as plt
from load_data import simulate


class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_layers=32):
        super(RNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn1 = torch.nn.RNN(input_size=self.input_size, num_layers=num_layers, hidden_size=self.hidden_layers,
                                 batch_first=True)
        self.rnn2 = torch.nn.RNN(input_size=self.hidden_layers, hidden_size=self.output_size, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_layers, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x, h_0):
        output, h = self.rnn1(x, h_0)
        # output, h = self.rnn2(output, h)
        output = self.linear(output)
        output = self.relu(output)
        return output, h


class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.lstm1 = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_layers, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=self.hidden_layers, hidden_size=self.hidden_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_layers, output_size)
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
            x = torch.flatten(x, 2)

            if dataloader.batch_size == 1:
                h_0 = torch.zeros((model.num_layers, model.hidden_layers))
            else:
                h_0 = torch.zeros((model.num_layers, dataloader.batch_size, model.hidden_layers))

            out, h_n = model(x[:, 0, :], h_0)

            for i in range(1, x.shape[1]):
                out, h_n = model(x[:, i, :], h_n)

            y = y.float()
            y = torch.flatten(y, 1)
            loss_train = loss_fn(out, y)  # calculate loss

            optimizer.zero_grad()  # set gradients to zero
            loss_train.backward()  # backwards pass
            optimizer.step()  # update model parameters

            if epoch == 1 or epoch % 5 == 0:
                print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")


def test_loop(model, loss, dataloader, num_nodes):
    actual_data = []
    predicted_data = []

    for time, (x, y) in enumerate(dataloader):
        for batch, (x, y) in enumerate(dataloader):

            x = x.float()
            x = torch.flatten(x, 2)

            if dataloader.batch_size == 1:
                h_0 = torch.zeros((model.num_layers, model.hidden_layers))
            else:
                h_0 = torch.zeros((model.num_layers, dataloader.batch_size, model.hidden_layers))

            out, h_n = model(x[:, 0, :], h_0)

            for i in range(1, x.shape[1]):
                out, h_n = model(x[:, i, :], h_n)

            y = y.float()

            out = torch.reshape(out, (2, 4))
            loss_test = loss(out, y[0])

            #if batch % 5 == 0:
                #print(f"Epoch {batch}, Testing loss {loss_test.item():.4f}")

        actual_data.append(y[0].detach().numpy())
        predicted_data.append(out.detach().numpy())

    graph_nodes(actual_data, predicted_data, num_nodes=num_nodes)


def graph_nodes(actual, predict, num_nodes):
    figure, axis = plt.subplots(num_nodes, 1)
    actual = np.asarray(actual)
    predict = np.asarray(predict)

    nodes_actual = [[] for i in range(num_nodes)]
    nodes_predict = [[] for i in range(num_nodes)]

    for i in range(num_nodes):
        nodes_actual[i].append(actual[:, i, :])
        nodes_predict[i].append(predict[:, i, :])
    nodes_actual = np.asarray(nodes_actual)
    nodes_predict = np.asarray(nodes_predict)

    for i in range(num_nodes):
        axis[i].plot(range(nodes_actual.shape[2]), nodes_actual[i][0])
        axis[i].plot(range(nodes_predict.shape[2]), nodes_predict[i][0], linestyle='dotted')
        print(nodes_predict[i][0])
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
    NODES = 2
    STEPS = 200
    PREVIOUS_STEPS = 20
    future_steps = 1
    stride = 1

    # sim = simulate(STEPS, NODES, file_name)

    NUM_EPOCHS = 50

    train_loader, test_loader = preprocess_as_temporal(file_name, STEPS, PREVIOUS_STEPS, future_steps=future_steps,
                                                       stride=stride)

    rnn = RNN(input_size=4 * NODES, output_size=4 * NODES, num_layers=2)
    lstm = LSTM(input_size=840, output_size=40)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()

    training_loop(NUM_EPOCHS, optimizer, rnn, loss, train_loader)
    # training_loop(NUM_EPOCHS, optimizer, lstm, loss, train_loader)
    test_loop(rnn, loss, test_loader, num_nodes=NODES)
    # test_loop(lstm, loss, test_loader)
