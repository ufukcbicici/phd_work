import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.routing_strategy.routing_strategy import RoutingStrategy


class ProbabilityThresholdedRoutingStrategy(RoutingStrategy):
    def __init__(self, probability_thresholds, **kwargs):
        super().__init__(**kwargs)
        self.probabilityThresholds = probability_thresholds

    def call(self, inputs, **kwargs):
        routing_probabilities = inputs[0]
        is_warm_up = inputs[1]
        block_id = inputs[2]
        training = kwargs["training"]

        probability_thresholds = self.probabilityThresholds[block_id]
        routing_matrix = tf.cast(tf.math.greater(routing_probabilities, probability_thresholds), dtype=tf.int32)
        return routing_matrix
