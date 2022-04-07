import tensorflow as tf
import numpy as np

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtDecisionLayer(tf.keras.layers.Layer):
    def __init__(self, use_boolean_mask_layer, node, decision_bn_momentum, next_block_path_count):
        super().__init__()
        with tf.name_scope("Node_{0}".format(node.index)):
            with tf.name_scope("decision_layer"):
                self.useBooleanMaskLayer = use_boolean_mask_layer
                self.cignNode = node
                self.nextBlockPathCount = next_block_path_count
                # self.infoGainLayer = InfoGainLayer(class_count=class_count)
                self.decisionBnMomentum = decision_bn_momentum
                # self.decisionBatchNorm = WeightedBatchNormalization(momentum=self.decisionBnMomentum, node=node)
                # self.decisionBatchNorm = tf.keras.layers.BatchNormalization(momentum=self.decisionBnMomentum)
                # self, momentum, epsilon, node = None, name = "", start_moving_averages_from_zero = False):
                if not use_boolean_mask_layer:
                    self.decisionBatchNorm = CigtBatchNormalization(momentum=self.decisionBnMomentum,
                                                                    epsilon=1e-3,
                                                                    node=self.cignNode,
                                                                    name="Node_{0}_cigt_batch_normalization_decision".
                                                                    format(self.cignNode.index),
                                                                    start_moving_averages_from_zero=False)
                else:
                    self.decisionBatchNorm = tf.keras.layers.BatchNormalization(momentum=self.decisionBnMomentum)
                if self.nextBlockPathCount > 1:
                    self.decisionActivationsLayer = CignDenseLayer(output_dim=self.nextBlockPathCount, activation=None,
                                                                   node=node, use_bias=True,
                                                                   name="Node_{0}_cigt_decision_hyperplane".format(
                                                                        self.cignNode.index))
                else:
                    self.decisionActivationsLayer = None
                # self.balanceCoeff = ig_balance_coefficient

    # @tf.function
    def call(self, inputs, **kwargs):
        h_net = inputs[0]
        labels = inputs[1]
        # temperature = inputs[2]
        training = kwargs["training"]
        if self.nextBlockPathCount > 1:
            if not self.useBooleanMaskLayer:
                # Apply Batch Normalization to inputs
                dummy_route_vector = tf.cast(tf.expand_dims(tf.ones_like(labels), axis=1), dtype=tf.int32)
                h_net_normed = self.decisionBatchNorm([h_net, dummy_route_vector], training=training)
            else:
                h_net_normed = self.decisionBatchNorm(h_net, training=training)
            activations = self.decisionActivationsLayer(h_net_normed)
        else:
            activations = tf.ones(shape=(tf.shape(h_net)[0], 1))
        return activations

        # ig_mask = tf.ones_like(labels)
        # ig_value, routing_probabilities = \
        #     self.infoGainLayer([activations, labels, temperature, self.balanceCoeff, ig_mask])
        # return ig_value, activations, routing_probabilities

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
