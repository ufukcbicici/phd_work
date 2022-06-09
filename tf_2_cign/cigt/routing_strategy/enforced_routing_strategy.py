import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.routing_strategy.routing_strategy import RoutingStrategy


class EnforcedRoutingStrategy(RoutingStrategy):
    def __init__(self, enforced_decision_vectors, **kwargs):
        super().__init__(**kwargs)
        self.enforcedDecisionVectors = enforced_decision_vectors

    def call(self, inputs, **kwargs):
        activation_matrix = inputs[0]
        is_warm_up = inputs[1]
        block_id = inputs[2]
        training = kwargs["training"]
        enforced_selections = tf.expand_dims(
            tf.cast(self.enforcedDecisionVectors[:, block_id], tf.bool), axis=1)

        path_count = tf.shape(activation_matrix)[1]
        all_paths_routing_matrix = tf.ones(shape=activation_matrix.shape, dtype=tf.int32)
        arg_max_routing_matrix = tf.one_hot(tf.argmax(activation_matrix, axis=1), path_count, dtype=tf.int32)
        routing_matrix = tf.where(enforced_selections, all_paths_routing_matrix, arg_max_routing_matrix)
        return routing_matrix
