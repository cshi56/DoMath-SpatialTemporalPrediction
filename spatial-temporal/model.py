import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Graph import *
import matplotlib.pyplot as plt
import typing

tf.random.set_seed(1234)


class GraphConv(layers.Layer):
    def __init__(self,
                 in_feat,
                 out_feat,
                 graph: Graph,
                 aggregation_type='mean',
                 combination_type='concat',
                 activation: typing.Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph = graph
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        initializer = keras.initializers.GlorotUniform()
        self.weight = tf.Variable(initial_value=initializer(shape=(in_feat, out_feat),
                                                            dtype='float32'),
                                  trainable=True)
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbor_representations: tf.Tensor):
        aggregation_func = {
            'sum': tf.math.unsorted_segment_sum,
            'mean': tf.math.unsorted_segment_mean,
            'max': tf.math.unsorted_segment_max
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbor_representations,
                self.graph.edges
            )

