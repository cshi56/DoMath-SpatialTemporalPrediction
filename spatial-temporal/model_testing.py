import torch
from RNNVectorized import RNNVectorized
from LSTMVectorized import LSTMVectorized
from GCRNN import GCRNN
from GCLSTM import GCLSTM

if __name__ == '__main__':
    DATA_FILE = 'data/fixed-parameters/150sims_50days_1nodes.npy'
    NODES = 1
    PREVIOUS_STEPS = 20
    FUTURE_STEPS = 1
    STRIDE = 5

    HIDDEN_SIZE = 128
    TRAIN_NUM = 100
    VAL_NUM = 50
    EPOCHS = 100
    INITIAL_LR = 0.001
    LR_DECAY = 0.9
    BATCH_SIZE = 300

    model = LSTMVectorized(NODES, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)

    model.train_model(DATA_FILE,
                      TRAIN_NUM,
                      VAL_NUM,
                      STRIDE,
                      EPOCHS,
                      lr=INITIAL_LR,
                      lr_decay=LR_DECAY,
                      optim=torch.optim.Adam,
                      batch_size=BATCH_SIZE)

    torch.save(model.state_dict(), 'models/1_nodes/veclstm128.pt')

    model.plot_loss(10)

    """
    nodes_list = [1, 2, 10, 20]
    datapath_list = ['data/200sims_50days_1nodes.npy',
                     'data/200sims_50days_2nodes.npy',
                     'data/200sims_50days_10nodes.npy',
                     'data/200sims_50days_20nodes.npy']

    for dex, nodes in enumerate(nodes_list):
        datapath = datapath_list[dex]

        model1 = RNNVectorized(nodes, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
        model2 = LSTMVectorized(nodes, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
        model3 = GCRNN(nodes, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)
        model4 = GCLSTM(nodes, 4, PREVIOUS_STEPS, FUTURE_STEPS, HIDDEN_SIZE)

        path1 = 'models/' + str(nodes) + '_nodes/rnn_vectorized_20prev_1fut.pt'
        path2 = 'models/' + str(nodes) + '_nodes/lstm_vectorized_20prev_1fut.pt'
        path3 = 'models/' + str(nodes) + '_nodes/gcrnn_20prev_1fut.pt'
        path4 = 'models/' + str(nodes) + '_nodes/gclstm_20prev_1fut.pt'

        models = [model1, model2, model3, model4]
        paths = [path1, path2, path3, path4]

        for i in range(len(models)):
            model = models[i]
            path = paths[i]

            model.train_model(datapath,
                              TRAIN_NUM,
                              VAL_NUM,
                              STRIDE,
                              EPOCHS,
                              optim=torch.optim.Adam,
                              lr=INITIAL_LR,
                              lr_decay=LR_DECAY)

            torch.save(model.state_dict(), path)
    """
