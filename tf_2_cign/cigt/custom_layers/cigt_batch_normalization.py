import numpy as np
import tensorflow as tf
import time

from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities.utilities import Utilities


# tf.autograph.set_verbosity(10, True)


class CigtBatchNormalization(WeightedBatchNormalization):
    def __init__(self, momentum, epsilon, node=None, name="", start_moving_averages_from_zero=False):
        super().__init__(momentum, node, name)
        # self.routingMatrix = node.routingMatrix
        self.epsilon = epsilon
        self.startMovingAveragesFromZero = start_moving_averages_from_zero

    # @tf.function
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        routing_matrix = inputs[1]
        is_training = kwargs["training"]
        num_routes = tf.shape(routing_matrix)[-1]
        route_width = tf.shape(x_)[-1] // num_routes
        population_count = tf.reduce_prod(tf.shape(x_)[1:-1])
        sample_counts_per_routes = tf.reduce_sum(routing_matrix, axis=0)
        route_probability_matrix = tf.cast(routing_matrix, dtype=tf.float32) * tf.expand_dims(1.0 / tf.cast(
            sample_counts_per_routes,
            dtype=tf.float32), axis=0)
        # Sample selection probabilities
        sample_probability_tensor = ((1.0 / tf.cast(population_count, tf.float32)) * tf.ones_like(x_))
        repeat_array = route_width * tf.ones_like(routing_matrix[0])
        mask_array = tf.repeat(route_probability_matrix, repeats=repeat_array, axis=-1)
        for i in range(len(x_.get_shape()) - 2):
            mask_array = tf.expand_dims(mask_array, axis=1)
        joint_probabilities_tensor = mask_array * sample_probability_tensor

        # Calculate batch mean, weighted
        weighted_x = joint_probabilities_tensor * x_
        mean_x = tf.reduce_sum(weighted_x, [idx for idx in range(len(x_.get_shape()) - 1)])

        # Calculate batch variance, weighted
        mean_x_expanded = tf.identity(mean_x)
        for idx in range(len(x_.get_shape()) - 1):
            mean_x_expanded = tf.expand_dims(mean_x_expanded, axis=0)
        zero_meaned = x_ - mean_x_expanded
        zero_meaned_squared = tf.square(zero_meaned)
        variance_x = tf.reduce_sum(joint_probabilities_tensor * zero_meaned_squared,
                                   [idx for idx in range(len(x_.get_shape()) - 1)])
        mu = mean_x
        sigma = variance_x
        if is_training:
            final_mean = mu
            final_var = sigma
        else:
            final_mean = self.popMean
            final_var = self.popVar
        normed_x = tf.nn.batch_normalization(x=x_,
                                             mean=final_mean,
                                             variance=final_var,
                                             offset=self.beta,
                                             scale=self.gamma,
                                             variance_epsilon=1e-5)
        if is_training:
            with tf.control_dependencies([normed_x]):
                if not self.startMovingAveragesFromZero:
                    new_pop_mean = tf.where(self.timesCalled > 0,
                                            (self.momentum * self.popMean + (1.0 - self.momentum) * mu), mu)
                    new_pop_var = tf.where(self.timesCalled > 0,
                                           (self.momentum * self.popVar + (1.0 - self.momentum) * sigma), sigma)
                else:
                    new_pop_mean = self.momentum * self.popMean + (1.0 - self.momentum) * mu
                    new_pop_var = self.momentum * self.popVar + (1.0 - self.momentum) * sigma

                self.timesCalled.assign_add(delta=1)
                self.popMean.assign(value=new_pop_mean)
                self.popVar.assign(value=new_pop_var)
        return normed_x
