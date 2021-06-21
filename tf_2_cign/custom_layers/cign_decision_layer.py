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
        self.nodeDegree = self.degreeList[node.depth]
        self.decisionBnMomentum = decision_bn_momentum
        self.decisionBatchNorm = WeightedBatchNormalization(momentum=self.decisionBnMomentum)
        self.hNet = self.nodeOutputsDict[node.index]["H"]
        self.decisionActivationsLayer = CignDenseLayer(output_dim=self.nodeDegree, activation=None,
                                                       node=node, use_bias=True, name="fc_op_decision")
        self.routingTemperature = self.routingTemperatures[self.cignNode.index]
        self.infoGainLayer = InfoGainLayer(class_count=self.network.classCount)
        self.labels = self.network.labels
        self.balanceCoeff = self.network.informationGainBalanceCoeff

    def call(self, inputs, **kwargs):
        ig_mask = inputs
        # Apply weighted batch norm to the h features
        h_net_normed = self.decisionBatchNorm([self.hNet, ig_mask])
        activations = self.decisionActivationsLayer(h_net_normed)
        ig_value = self.infoGainLayer([activations, self.labels, self.routingTemperature, self.balanceCoeff, ig_mask])
        # Information gain based routing matrix
        ig_routing_matrix = tf.one_hot(tf.argmax(activations, axis=1), self.nodeDegree, dtype=tf.int32)
        mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
        output_ig_routing_matrix = tf.cast(
            tf.logical_and(tf.cast(ig_routing_matrix, dtype=tf.bool), tf.cast(mask_as_matrix, dtype=tf.bool)),
            dtype=tf.int32)
        return h_net_normed, ig_value, output_ig_routing_matrix
