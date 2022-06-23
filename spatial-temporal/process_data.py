import torch
import numpy as np
import random
from torch_geometric.data import Data

random.seed(1234)
torch.manual_seed(1234)


def generate_dataset(file, edge_index, sequence_length, stride):
    graph_data = np.load(file)
    sim_length = len(graph_data[0])
    populations = [sum(graph_data[i][0]) for i in range(len(graph_data))]
    graph_data = graph_data / max(populations)

    graph_series = []
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    for time in range(sim_length):
        vertices = []
        for node_index in range(len(graph_data)):
            seir = graph_data[node_index][time]
            vertices.append(seir)
        vertices = np.asarray(vertices)
        vertices = torch.tensor(vertices, dtype=torch.float)
        graph = Data(vertices, edge_index)
        graph_series.append(graph)

    graph_training_data = []

    for index in range(0, sim_length - sequence_length, stride):
        datum = graph_series[index: index + sequence_length]
        label = graph_series[index + sequence_length].x
        graph_training_data.append((datum, label))

    return graph_training_data
