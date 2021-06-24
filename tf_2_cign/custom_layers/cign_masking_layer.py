import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities import Utilities


class CignMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, network, node):
        super().__init__()
        self.network = network
        self.cignNode = node

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        h_input = inputs[1]
        parent_ig_matrix = inputs[2]
        parent_sc_matrix = inputs[3]
        sibling_index = inputs[4]

        ig_mask = parent_ig_matrix[:, sibling_index]
        sc_mask = parent_sc_matrix[:, sibling_index]
        sample_count = tf.reduce_sum(tf.cast(sc_mask, tf.float32))
        is_node_open = tf.greater(sample_count, 0.0)
        return f_input, h_input, ig_mask, sc_mask, sample_count, is_node_open
