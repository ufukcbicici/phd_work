import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.routing_strategy.routing_strategy import RoutingStrategy


class ApproximateTrainingStrategy(RoutingStrategy):
    def __init__(self, warm_up_epoch_count, **kwargs):
        super().__init__(**kwargs)
        self.warmUpEpochCount = warm_up_epoch_count
        self.isInWarmUp = True
        self.warmUpFinalIteration = None

    def set_training_statistics(self, iteration_count, epoch_count):
        self.iterationCount = iteration_count
        self.epochCount = epoch_count
        if self.epochCount > self.warmUpEpochCount:
            if self.isInWarmUp:
                self.warmUpFinalIteration = self.iterationCount
            self.isInWarmUp = False

    def modify_temperature(self, softmax_decay_controller):
        if not self.isInWarmUp:
            decay_t = self.iterationCount - self.warmUpFinalIteration
            softmax_decay_controller.update(iteration=decay_t)

    def calculate_information_gain_losses(self, ig_losses, decision_loss_coefficient):
        if self.isInWarmUp:
            loss = 0.0 * tf.add_n(ig_losses)
        else:
            loss = decision_loss_coefficient * tf.add_n(ig_losses)
        return loss

    def call(self, inputs, **kwargs):
        activation_matrix = inputs
        training = kwargs["training"]

        path_count = tf.shape(activation_matrix)[1]

        if self.isInWarmUp:
            routing_matrix = tf.ones(shape=activation_matrix.shape, dtype=tf.int32)
        else:
            routing_matrix = tf.one_hot(tf.argmax(activation_matrix, axis=1), path_count, dtype=tf.int32)
        return routing_matrix
