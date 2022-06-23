import torch
from torch import nn
import numpy as np
import random
import torch_geometric as pyg
from process_data import generate_dataset
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from torch_model import ConvRNN

random.seed(1234)
torch.manual_seed(1234)


if __name__ == '__main__':
    model = ConvRNN(8, 4, 74)
    model.load_state_dict(torch.load('models/conv_rnn1.pt'))


