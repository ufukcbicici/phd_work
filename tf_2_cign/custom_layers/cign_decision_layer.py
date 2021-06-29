import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities import Utilities


class CignDecisionLayer(tf.keras.layers.Layer):
    def __init__(self, network, node, decision_bn_momentum):
        super().__init__()
        self.network = network
        self.cignNode = node
        self.nodeDegree = self.network.degreeList[node.depth]
        self.infoGainLayer = InfoGainLayer(class_count=self.network.classCount)
        self.decisionBnMomentum = decision_bn_momentum
        self.decisionBatchNorm = WeightedBatchNormalization(momentum=self.decisionBnMomentum)
        self.decisionActivationsLayer = CignDenseLayer(output_dim=self.nodeDegree, activation=None,
                                                       node=node, use_bias=True, name="fc_op_decision")
        self.balanceCoeff = self.network.informationGainBalanceCoeff

    def call(self, inputs, **kwargs):
        h_net = inputs[0]
        ig_mask = inputs[1]
        labels = inputs[2]
        temperature = inputs[3]

        # Apply weighted batch norm to the h features
        h_net_normed = self.decisionBatchNorm([h_net, ig_mask])
        activations = self.decisionActivationsLayer(h_net_normed)
        ig_value = self.infoGainLayer([activations, labels, temperature, self.balanceCoeff, ig_mask])
        # Information gain based routing matrix
        ig_routing_matrix = tf.one_hot(tf.argmax(activations, axis=1), self.nodeDegree, dtype=tf.int32)
        mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
        output_ig_routing_matrix = tf.cast(
            tf.logical_and(tf.cast(ig_routing_matrix, dtype=tf.bool), tf.cast(mask_as_matrix, dtype=tf.bool)),
            dtype=tf.int32)
        return h_net_normed, ig_value, output_ig_routing_matrix, activations
