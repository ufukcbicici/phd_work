import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities import Utilities


class CignVanillaScRoutingLayer(tf.keras.layers.Layer):
    def __init__(self, network):
        super().__init__()
        self.network = network
        # sself.level = level

    @tf.function
    def call(self, inputs, **kwargs):
        input_f_tensor = inputs[0]
        input_ig_routing_matrix = inputs[1]

        secondary_routing_matrix = tf.identity(input_ig_routing_matrix)
        return secondary_routing_matrix
