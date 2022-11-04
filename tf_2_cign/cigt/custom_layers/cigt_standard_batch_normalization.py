import numpy as np
import tensorflow as tf
import time

from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities.utilities import Utilities


# tf.autograph.set_verbosity(10, True)


class CigtStandardBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, momentum, epsilon, node, name):
        super(CigtStandardBatchNormalization, self).__init__(name=name)
        self.node = node
        self.batchNorm = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)

    # @tf.function
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        routing_matrix = inputs[1]
        is_training = kwargs["training"]
        normed_x = self.batchNorm(x_, training=is_training)
        return normed_x
