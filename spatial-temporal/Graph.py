import numpy as np
from simulation import Simulation


class Graph:
    def __init__(self, vertices, edges, weights):
        self.vertices = vertices
        self.edges = edges
        self.weights = weights

    def __iter__(self):
        yield from self.vertices

    def __getitem__(self, i):
        return self.vertices[i]

    def is_edge(self, i, j):
        return self.edges[i][j]

    def edge_weight(self, i, j):
        return self.weights[i][j]


class TimeSeriesGraph:
    def __init__(self, graphs):
        self.graphs = graphs
        self.length = len(graphs)

    def __iter__(self):
        yield from self.graphs

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return self.length


def graph_from_simulation(simulation):
    length = len(simulation)
    edges = simulation.diffusion_matrix
    weights = simulation.diffusion_matrix
    graphs = []
    for time in range(length):
        vertices = []
        for node in simulation.nodes:
            seir_value = node[time]
            vertices.append(seir_value)
        graph = Graph(vertices, edges, weights)
        graphs.append(graph)
    time_series_graph = TimeSeriesGraph(graphs)
    return time_series_graph

