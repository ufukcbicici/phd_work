import tensorflow as tf
import numpy as np

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class CigtGumbelSoftmax(tf.keras.layers.Layer):
    def __int__(self):
        super(CigtGumbelSoftmax, self).__int__()

    # def sample_from_gumbel_softmax(self, logits, temperature, z_sample_count, eps=1e-20):
    #     logits_shape = tf.shape(logits)
    #     samples_shape = tf.concat([logits_shape[0], logits_shape[1], z_sample_count], axis=0)
    #     U_ = tf.random.uniform(shape=samples_shape, minval=0.0, maxval=1.0)
    #     G_ = -tf.math.log(-tf.math.log(U_ + eps) + eps)
    #     log_logits = tf.math.log(logits + eps)
    #     log_logits = tf.expand_dims(log_logits, axis=-1)
    #     gumbel_logits = log_logits + G_
    #     gumbel_logits = gumbel_logits / temperature
    #     z_samples = tf.nn.softmax(gumbel_logits)
    #     return z_samples

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
        eps = 1e-20
        logits = inputs[0]
        temperature = inputs[1]
        z_sample_count = inputs[2]
        training = kwargs["training"]

        logits_shape = tf.shape(logits)
        samples_shape = tf.concat([logits_shape[0], logits_shape[1], z_sample_count], axis=0)
        U_ = tf.random.uniform(shape=samples_shape, minval=0.0, maxval=1.0)
        G_ = -tf.math.log(-tf.math.log(U_ + eps) + eps)
        log_logits = tf.math.log(logits + eps)
        log_logits = tf.expand_dims(log_logits, axis=-1)
        gumbel_logits = log_logits + G_
        gumbel_logits_tempered = gumbel_logits / temperature
        z_samples = tf.nn.softmax(gumbel_logits_tempered, axis=1)
        return z_samples, gumbel_logits, gumbel_logits_tempered, log_logits, G_
