import torch
from RNNVectorized import RNNVectorized
from LSTMVectorized import LSTMVectorized
from GCRNN import GCRNN


if __name__ == '__main__':
    DATA_FILE = 'data/200sims_50days_10nodes.npy'
    SAVE_PATH = 'models/1_node/rnn_vectorized_20prev_1fut'
    NODES = 10
    PREVIOUS_STEPS = 20
    FUTURE_STEPS = 1
    STRIDE = 5

    HIDDEN_SIZE = 64
    TRAIN_NUM = 100
    VAL_NUM = 50
    EPOCHS = 50
    INITIAL_LR = 0.001
    LR_DECAY = 0.9

    model1 = RNNVectorized(NODES, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
    model2 = LSTMVectorized(NODES, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
    model3 = GCRNN(NODES, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)

    path1 = 'models/10_nodes/rnn_vectorized_20prev_1fut'
    path2 = 'models/10_nodes/lstm_vectorized_20prev_1fut'
    path3 = 'models/10_nodes/gcrnn_20prev_1fut'

    models, paths = [model1, model2, model3], [path1, path2, path3]

    for i in range(3):
        model = models[i]
        path = paths[i]

        model.train_model(DATA_FILE,
                          TRAIN_NUM,
                          VAL_NUM,
                          STRIDE,
                          EPOCHS,
                          optim=torch.optim.Adam,
                          lr=INITIAL_LR,
                          lr_decay=LR_DECAY)

        torch.save(model.state_dict(), path)

        model.plot_loss(20)
