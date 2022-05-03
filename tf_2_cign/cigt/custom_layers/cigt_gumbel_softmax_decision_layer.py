import tensorflow as tf
import numpy as np

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtGumbelSoftmaxDecisionLayer(CigtDecisionLayer):
    def __init__(self, node, decision_bn_momentum, next_block_path_count, class_count, ig_balance_coefficient):
        super().__init__(node, decision_bn_momentum, next_block_path_count, class_count, ig_balance_coefficient)

#
#     # @tf.function
    def call(self, inputs, **kwargs):
        h_net = inputs[0]
        labels = inputs[1]
        temperature = inputs[2]
        training = kwargs["training"]
    
        # Apply Batch Normalization to inputs
        dummy_route_vector = tf.cast(tf.expand_dims(tf.ones_like(labels), axis=1), dtype=tf.int32)
        h_net_normed = self.decisionBatchNorm([h_net, dummy_route_vector], training=training)
        activations = self.decisionActivationsLayer(h_net_normed)
#         ig_mask = tf.ones_like(labels)
#         ig_value, routing_probabilities = \
#             self.infoGainLayer([activations, labels, temperature, self.balanceCoeff, ig_mask])
#         return ig_value, activations, routing_probabilities
#
#         # ig_mask = inputs[1]
#         # labels = inputs[2]
#         # temperature = inputs[3]
#         #
#         # # Apply weighted batch norm to the h features
#         # h_net_normed = self.decisionBatchNorm([h_net, ig_mask])
#         # activations = self.decisionActivationsLayer(h_net_normed)
#         # ig_value = self.infoGainLayer([activations, labels, temperature, self.balanceCoeff, ig_mask])
#         # # Information gain based routing matrix
#         # ig_routing_matrix = tf.one_hot(tf.argmax(activations, axis=1), self.nodeDegree, dtype=tf.int32)
#         # mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
#         # output_ig_routing_matrix = tf.cast(
#         #     tf.logical_and(tf.cast(ig_routing_matrix, dtype=tf.bool), tf.cast(mask_as_matrix, dtype=tf.bool)),
#         #     dtype=tf.int32)
#         # return h_net_normed, ig_value, output_ig_routing_matrix, activations
