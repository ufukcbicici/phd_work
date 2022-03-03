import tensorflow as tf
import numpy as np

from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtDecisionLayer(tf.keras.layers.Layer):
    def __init__(self, network, node, decision_bn_momentum):
        super().__init__()
        with tf.name_scope("Node_{0}".format(node.index)):
            with tf.name_scope("decision_layer"):
                self.network = network
                self.cignNode = node
                self.nodeDegree = self.network.degreeList[node.depth]
                self.infoGainLayer = InfoGainLayer(class_count=self.network.classCount)
                self.decisionBnMomentum = decision_bn_momentum
                # self.decisionBatchNorm = WeightedBatchNormalization(momentum=self.decisionBnMomentum, node=node)
                self.decisionBatchNorm = tf.keras.layers.BatchNormalization(momentum=self.decisionBnMomentum)
                self.decisionActivationsLayer = CignDenseLayer(output_dim=self.nodeDegree, activation=None,
                                                               node=node, use_bias=True, name="fc_op_decision")
                self.balanceCoeff = self.network.informationGainBalanceCoeff

    # @tf.function
    def call(self, inputs, **kwargs):
        h_net = inputs[0]
        labels = inputs[1]
        temperature = inputs[2]

        # Apply Batch Normalization to inputs
        h_net_normed = self.decisionBatchNorm(h_net)
        activations = self.decisionActivationsLayer(h_net_normed)
        ig_mask = tf.ones_like(labels)
        ig_value = self.infoGainLayer([activations, labels, temperature, self.balanceCoeff, ig_mask])
        return ig_value, activations

        # ig_mask = inputs[1]
        # labels = inputs[2]
        # temperature = inputs[3]
        #
        # # Apply weighted batch norm to the h features
        # h_net_normed = self.decisionBatchNorm([h_net, ig_mask])
        # activations = self.decisionActivationsLayer(h_net_normed)
        # ig_value = self.infoGainLayer([activations, labels, temperature, self.balanceCoeff, ig_mask])
        # # Information gain based routing matrix
        # ig_routing_matrix = tf.one_hot(tf.argmax(activations, axis=1), self.nodeDegree, dtype=tf.int32)
        # mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
        # output_ig_routing_matrix = tf.cast(
        #     tf.logical_and(tf.cast(ig_routing_matrix, dtype=tf.bool), tf.cast(mask_as_matrix, dtype=tf.bool)),
        #     dtype=tf.int32)
        # return h_net_normed, ig_value, output_ig_routing_matrix, activations
