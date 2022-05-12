import numpy as np
import tensorflow as tf
import time

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.cigt.custom_layers.cigt_unmasking_layer import CigtUnmaskingLayer
from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities.utilities import Utilities


# tf.autograph.set_verbosity(10, True)


class CigtProbabilisticBatchNormalization(WeightedBatchNormalization):
    def __init__(self, momentum, epsilon, node=None, name="",
                 start_moving_averages_from_zero=False, normalize_routing_matrix=True):
        super().__init__(momentum, node, name)
        # self.routingMatrix = node.routingMatrix
        self.epsilon = epsilon
        self.batchMean = None
        self.batchVar = None
        # self.batchJointProbability = None
        self.startMovingAveragesFromZero = start_moving_averages_from_zero
        self.normalizeRoutingMatrix = normalize_routing_matrix
        self.unmaskLayer = CigtUnmaskingLayer()
        self.maskLayer = CigtMaskingLayer()

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        input_dim = input_shape[0].as_list()[-1]

        self.batchMean = self.add_weight(
            name="batchMean" if self.node is None else Utilities.get_variable_name(
                name="{0}_batchMean".format(self.opPrefix),
                node=self.node),
            shape=input_dim,
            initializer=tf.keras.initializers.Constant(value=0.0),
            trainable=False)

        self.batchVar = self.add_weight(
            name="batchVar" if self.node is None else Utilities.get_variable_name(
                name="{0}_batchVar".format(self.opPrefix),
                node=self.node),
            shape=input_dim,
            initializer=tf.keras.initializers.Constant(value=0.0),
            trainable=False)

        # self.batchJointProbability = self.add_weight(
        #     name="batchJointProbability" if self.node is None else Utilities.get_variable_name(
        #         name="{0}_batchJointProbability".format(self.opPrefix),
        #         node=self.node),
        #     shape=input_shape,
        #     initializer=tf.keras.initializers.Constant(value=0.0),
        #     trainable=False)

    # @tf.function
    def calculate_joint_probabilities(self, x_, routing_matrix):
        num_routes = tf.shape(routing_matrix)[-1]
        route_width = tf.shape(x_)[-1] // num_routes
        batch_size = tf.cast(tf.shape(x_)[0], dtype=tf.float32)
        population_count = tf.cast(tf.reduce_prod(tf.shape(x_)[1:-1]), dtype=tf.float32)
        joint_probabilities_s_r_ch_c = tf.ones_like(x_)
        # Build the joint probability matrix
        # Step 1): Sample distribution: p(s)
        joint_probabilities_s_r_ch_c = joint_probabilities_s_r_ch_c * (1.0 / batch_size)
        # Step 2): Build and extend route distribution p(r|s)
        repeat_array = route_width * tf.ones_like(routing_matrix[0], dtype=tf.int32)
        route_probabilities = tf.repeat(routing_matrix, repeats=repeat_array, axis=-1)
        for i in range(len(x_.get_shape()) - 2):
            route_probabilities = tf.expand_dims(route_probabilities, axis=1)
        joint_probabilities_s_r_ch_c = joint_probabilities_s_r_ch_c * tf.cast(route_probabilities, dtype=tf.float32)
        # Step 3): Calculate channel probabilities given a route p(ch|r) = (1.0 / route_width)
        joint_probabilities_s_r_ch_c = (1.0 / tf.cast(route_width, dtype=tf.float32)) * joint_probabilities_s_r_ch_c
        # Step 4): Probability of an entry in the feature map p(c|ch) = (1.0 / population_count)
        joint_probabilities_s_r_ch_c = (1.0 / population_count) * joint_probabilities_s_r_ch_c
        marginal_probabilities_r_ch = tf.reduce_sum(joint_probabilities_s_r_ch_c,
                                                    [idx for idx in range(len(x_.get_shape()) - 1)])
        return joint_probabilities_s_r_ch_c, marginal_probabilities_r_ch

    # @tf.function
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        routing_matrix_unnormed = inputs[1]
        is_training = kwargs["training"]

        # Unscale the input
        x_ = self.unmaskLayer([x_, routing_matrix_unnormed])

        # Normalize the routing matrix
        if self.normalizeRoutingMatrix:
            normalizing_constants = tf.reduce_sum(routing_matrix_unnormed, axis=1, keepdims=True)
            routing_matrix = routing_matrix_unnormed * tf.math.reciprocal_no_nan(normalizing_constants)
        else:
            routing_matrix = tf.identity(routing_matrix_unnormed)

        joint_probabilities_s_r_ch_c, marginal_probabilities_r_ch = self.calculate_joint_probabilities(
            x_=x_, routing_matrix=routing_matrix)
        for i in range(len(x_.get_shape()) - 1):
            marginal_probabilities_r_ch = tf.expand_dims(marginal_probabilities_r_ch, axis=0)
        conditional_probabilities_s_c_given_ch_r = \
            joint_probabilities_s_r_ch_c * tf.math.reciprocal(marginal_probabilities_r_ch)
        # Calculate batch mean, weighted
        weighted_x = conditional_probabilities_s_c_given_ch_r * x_
        mean_x = tf.reduce_sum(weighted_x, [idx for idx in range(len(x_.get_shape()) - 1)])
        # Calculate batch variance, weighted
        mean_x_expanded = tf.identity(mean_x)
        for idx in range(len(x_.get_shape()) - 1):
            mean_x_expanded = tf.expand_dims(mean_x_expanded, axis=0)
        zero_meaned = x_ - mean_x_expanded
        zero_meaned_squared = tf.square(zero_meaned)
        variance_x = tf.reduce_sum(conditional_probabilities_s_c_given_ch_r * zero_meaned_squared,
                                   [idx for idx in range(len(x_.get_shape()) - 1)])
        mu = mean_x
        sigma = variance_x
        self.batchMean.assign(value=mu)
        self.batchVar.assign(value=sigma)
        if is_training:
            final_mean = mu
            final_var = sigma
            # tf.print("!!! CIGT - TRAINING!!!")
        else:
            final_mean = self.popMean
            final_var = self.popVar
            # tf.print("!!! CIGT - TEST!!!")
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
        # return normed_x, mean_x, variance_x, weighted_x, \
        #        joint_probabilities_s_r_ch_c, marginal_probabilities_r_ch, zero_meaned

        # Scale the input back to original
        normed_x = self.maskLayer([normed_x, routing_matrix_unnormed])
        return normed_x, joint_probabilities_s_r_ch_c
