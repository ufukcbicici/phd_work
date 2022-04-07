import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.routing_strategy.routing_strategy import RoutingStrategy


class ApproximateTrainingStrategyWithRandomRouting(RoutingStrategy):
    def __init__(self, use_gumbel, **kwargs):
        super().__init__(**kwargs)
        self.useGumbel = use_gumbel

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        return -tf.math.log(-tf.math.log(tf.random.uniform(shape, minval=0, maxval=1) + eps) + eps)

    def call(self, inputs, **kwargs):
        ig_activations_matrix = inputs[0]
        is_warm_up = inputs[1]
        temperature = inputs[2]
        training = kwargs["training"]
        ig_activations_matrix_tempered = ig_activations_matrix / temperature

        path_count = tf.shape(ig_activations_matrix_tempered)[1]
        if self.useGumbel:
            noise_matrix = self.sample_gumbel(tf.shape(ig_activations_matrix_tempered))
        else:
            noise_matrix = tf.zeros_like(ig_activations_matrix_tempered)

        routing_activations_final = tf.where(training, ig_activations_matrix_tempered + noise_matrix,
                                             ig_activations_matrix_tempered)

        warm_up_matrix = tf.random.uniform(shape=tf.shape(ig_activations_matrix_tempered))
        random_routing_matrix = tf.one_hot(tf.argmax(warm_up_matrix, axis=1), path_count, dtype=tf.int32)
        arg_max_routing_matrix = tf.one_hot(tf.argmax(routing_activations_final, axis=1), path_count, dtype=tf.int32)
        routing_matrix = tf.where(is_warm_up, random_routing_matrix, arg_max_routing_matrix)
        return routing_matrix

