import tensorflow as tf
import numpy as np

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_gumbel_softmax import CigtGumbelSoftmax
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtGumbelSoftmaxDecisionLayer(CigtDecisionLayer):
    def __init__(self, node, decision_bn_momentum, next_block_path_count, class_count, ig_balance_coefficient,
                 sample_count=250):
        super().__init__(node, decision_bn_momentum, next_block_path_count,
                         class_count, ig_balance_coefficient, from_logits=False)
        self.softPlusLayer = tf.keras.activations.softplus()
        self.gsLayer = CigtGumbelSoftmax()
        self.sampleCount = sample_count

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
        ig_mask = tf.ones_like(labels)
        logits = self.softPlusLayer(activations)
        z_samples = self.gsLayer([logits, temperature, self.sampleCount], training=training)
        z_expected = tf.reduce_mean(z_samples, axis=-1)
        ig_value, _ = self.infoGainLayer([z_expected, labels, 1.0, self.balanceCoeff, ig_mask])
        routing_probabilities = self.gsLayer([logits, temperature, 1], training=training)
        routing_probabilities = tf.squeeze(routing_probabilities)
        return ig_value, z_expected, routing_probabilities
