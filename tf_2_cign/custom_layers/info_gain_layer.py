import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer


class InfoGainLayer(tf.keras.layers.Layer):
    def __init__(self, class_count, normalize):
        super().__init__()
        self.classCount = tf.constant(class_count)
        self.normalize = normalize

    # @tf.function
    def call(self, inputs, **kwargs):
        activations = inputs[0]
        labels = inputs[1]
        temperature = inputs[2]
        balance_coefficient = inputs[3]
        weight_vector = inputs[4]
        # probability_vector = tf.cast(weight_vector / tf.reduce_sum(weight_vector), dtype=activations.dtype)
        sample_count = tf.reduce_sum(weight_vector)
        probability_vector = tf.math.divide_no_nan(tf.cast(weight_vector, dtype=activations.dtype),
                                                   tf.cast(sample_count, dtype=activations.dtype))

        batch_size = tf.shape(activations)[0]
        node_degree = tf.shape(activations)[1]

        # self.max_information_gain = (self.balance_coefficient * math.log(self.num_routes)) + math.log(self.num_classes)
        # self.min_information_gain = - math.log(self.num_classes * self.num_routes)

        A = tf.cast(node_degree, dtype=tf.float32)
        B = tf.cast(self.classCount, dtype=tf.float32)
        max_information_gain = (balance_coefficient * tf.math.log(A)) + tf.math.log(B)
        min_information_gain = - tf.math.log(A * B)

        joint_distribution = tf.ones(shape=(batch_size, self.classCount, node_degree), dtype=activations.dtype)

        # Calculate p(x)
        joint_distribution = joint_distribution * tf.expand_dims(tf.expand_dims(probability_vector, axis=-1), axis=-1)

        # Calculate p(c|x) * p(x) = p(x,c)
        p_c_given_x = tf.one_hot(labels, self.classCount)
        joint_distribution = joint_distribution * tf.expand_dims(p_c_given_x, axis=2)

        # Calculate p(n|x,c) * p(x,c) = p(x,c,n). Note that p(n|x,c) = p(n|x) since we assume conditional independence
        activations_with_temperature = activations / temperature
        p_n_given_x = tf.nn.softmax(activations_with_temperature)
        p_xcn = joint_distribution * tf.expand_dims(p_n_given_x, axis=1)

        # Calculate p(c,n)
        marginal_p_cn = tf.reduce_sum(p_xcn, axis=0)
        # Calculate p(n)
        marginal_p_n = tf.reduce_sum(marginal_p_cn, axis=0)
        # Calculate p(c)
        marginal_p_c = tf.reduce_sum(marginal_p_cn, axis=1)
        # Calculate entropies
        entropy_p_cn, log_prob_p_cn = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_cn)
        entropy_p_n, log_prob_p_n = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_n)
        entropy_p_c, log_prob_p_c = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_c)
        # Calculate the information gain
        information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
        if self.normalize:
            information_gain = 1.0 - (information_gain - min_information_gain) / \
                               (max_information_gain - min_information_gain)
        else:
            information_gain = -1.0 * information_gain
        return information_gain, p_n_given_x
