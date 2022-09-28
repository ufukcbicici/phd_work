import tensorflow as tf
import numpy as np

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_gumbel_softmax import CigtGumbelSoftmax
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtRandomDecisionLayer(CigtDecisionLayer):
    def __init__(self, node, decision_bn_momentum, next_block_path_count, class_count, ig_balance_coefficient):
        super().__init__(node, decision_bn_momentum, next_block_path_count,
                         class_count, ig_balance_coefficient, from_logits=False)

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

        random_activations = tf.random.uniform(shape=tf.shape(activations), dtype=activations.dtype)
        return 0.0, random_activations, random_activations, random_activations, h_net_normed
