import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.routing_strategy.routing_strategy import RoutingStrategy


class ApproximateTrainingStrategy(RoutingStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        ig_activations_matrix = inputs[0]
        is_warm_up = inputs[1]
        temperature = inputs[2]
        training = kwargs["training"]

        path_count = tf.shape(ig_activations_matrix)[1]

        warm_up_matrix = tf.ones(shape=ig_activations_matrix.shape, dtype=tf.int32)
        arg_max_routing_matrix = tf.one_hot(tf.argmax(ig_activations_matrix, axis=1), path_count, dtype=tf.int32)

        routing_matrix = tf.where(is_warm_up, warm_up_matrix, arg_max_routing_matrix)
        return routing_matrix

