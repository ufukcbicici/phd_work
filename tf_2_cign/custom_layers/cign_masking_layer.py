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
        if node.isRoot:
            self.parentNode = None
            self.siblingIndex = 0
            self.parentIgMaskMatrix = tf.expand_dims(tf.ones_like(self.network.labels), axis=1)
            self.parentScMaskMatrix = tf.expand_dims(tf.ones_like(self.network.labels), axis=1)
            self.parent_F = self.network.inputs
            self.parent_H = None
        else:
            self.parentNode = self.network.dagObject.parents(node=node)[0]
            self.siblingIndex = self.network.get_node_sibling_index(node=node)
            self.parentIgMaskMatrix = self.network.nodeOutputsDict[self.parentNode.index]["ig_mask_matrix"]
            self.parentScMaskMatrix = self.network.nodeOutputsDict[self.parentNode.index]["secondary_mask_matrix"]
            self.parent_F = self.network.nodeOutputsDict[self.parentNode.index]["F"]
            self.parent_H = self.network.nodeOutputsDict[self.parentNode.index]["H"]

    def call(self, inputs, **kwargs):
        f_input = self.parent_F
        h_input = self.parent_H
        ig_mask = tf.identity(self.parentIgMaskMatrix[:, self.siblingIndex])
        sc_mask = tf.identity(self.parentScMaskMatrix[:, self.siblingIndex])
        sample_count = tf.reduce_sum(tf.cast(sc_mask, tf.float32))
        is_node_open = tf.greater(sample_count, 0.0)
        return f_input, h_input, ig_mask, sc_mask, sample_count, is_node_open
