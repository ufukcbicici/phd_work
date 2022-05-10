import tensorflow as tf
import numpy as np

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtGumbelSoftmaxDecisionLayer(CigtDecisionLayer):
    def __init__(self, node, decision_bn_momentum, next_block_path_count, class_count, ig_balance_coefficient,
                 sample_count=100):
        super().__init__(node, decision_bn_momentum, next_block_path_count, class_count, ig_balance_coefficient)
        self.softPlusLayer = tf.keras.activations.softplus()

    def sample_from_gumbel_softmax(self, logits, temperature, z_sample_count, eps=1e-20):
        logits_shape = tf.shape(logits)
        samples_shape = tf.concat([logits_shape[0], logits_shape[1], z_sample_count], axis=0)
        U_ = tf.random.uniform(shape=samples_shape, minval=0.0, maxval=1.0)
        G_ = -tf.math.log(-tf.math.log(U_ + eps) + eps)
        log_logits = tf.math.log(logits + eps)
        log_logits = tf.expand_dims(log_logits, axis=-1)
        gumbel_logits = log_logits + G_
        gumbel_logits = gumbel_logits / temperature
        z_samples = tf.nn.softmax(gumbel_logits)
        return z_samples

        # Gumbel-Softmax

#         log_logits = tf.math.log(logits)
#         log_logits = tf.expand_dims(log_logits, axis=-1)
#         pre_transform_G = log_logits + G_
#         gumbel_logits = pre_transform_G / temperature
#         # gumbel_logits = tf.math.exp(pre_transform_G_temp_divided)
#         z_samples_softmax = tf.nn.softmax(gumbel_logits)
#
#         if not use_stable_version:
#             exp_logits = tf.math.exp(gumbel_logits)
#             nominator = tf.expand_dims(tf.reduce_sum(exp_logits, axis=1), axis=1)
#             z_samples = exp_logits / nominator
#         else:
#             # Numerically stable
#             log_sum_exp = tf.expand_dims(tf.reduce_logsumexp(gumbel_logits, axis=1), axis=1)
#             log_z_samples = gumbel_logits - log_sum_exp
#             z_samples = tf.math.exp(log_z_samples)
#         return z_samples
# #
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
        # Softplus, because the logits must be positive for Gumbel-Softmax to work correctly.
        logits = self.softPlusLayer(activations)



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
