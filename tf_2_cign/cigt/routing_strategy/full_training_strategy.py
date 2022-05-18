import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.routing_strategy.routing_strategy import RoutingStrategy


class FullTrainingStrategy(RoutingStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        soft_routing_matrix = inputs[0]
        is_warm_up = inputs[1]
        training = kwargs["training"]

        path_count = tf.shape(soft_routing_matrix)[1]

        warm_up_matrix = tf.ones(shape=soft_routing_matrix.shape, dtype=soft_routing_matrix.dtype)
        routing_matrix = tf.where(is_warm_up, warm_up_matrix, soft_routing_matrix)
        # hard_routing_matrix = tf.one_hot(tf.argmax(soft_routing_matrix, axis=1), path_count,
        #                                  dtype=soft_routing_matrix.dtype)
        # after_warm_up_matrix = tf.where(training, soft_routing_matrix, hard_routing_matrix)
        # routing_matrix = tf.where(is_warm_up, warm_up_matrix, after_warm_up_matrix)
        return routing_matrix

