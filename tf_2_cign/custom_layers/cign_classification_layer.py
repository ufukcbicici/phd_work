import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities import Utilities


class CignClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, network, node, class_count):
        super().__init__()
        with tf.name_scope("Node_{0}".format(node.index)):
            with tf.name_scope("classification_layer"):
                self.network = network
                self.cignNode = node
                self.classCount = class_count
                self.lossLayer = CignDenseLayer(output_dim=self.classCount, activation=None, node=node, use_bias=True,
                                                name="loss_layer")

    def call(self, inputs, **kwargs):
        f_net = inputs[0]
        sc_mask = inputs[1]
        labels = inputs[2]

        logits = self.lossLayer(f_net)
        posteriors = tf.nn.softmax(logits)
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Convert sc_mask to weight vector
        weight_vector = sc_mask
        sample_count = tf.reduce_sum(weight_vector)

        probability_vector = tf.math.divide_no_nan(tf.cast(weight_vector, dtype=cross_entropy_loss_tensor.dtype),
                                                   tf.cast(sample_count, dtype=cross_entropy_loss_tensor.dtype))

        # Weight losses
        weighted_losses = probability_vector * cross_entropy_loss_tensor
        loss = tf.reduce_sum(weighted_losses)

        return cross_entropy_loss_tensor, probability_vector, weighted_losses, loss, posteriors
