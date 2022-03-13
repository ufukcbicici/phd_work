import numpy as np
import tensorflow as tf


class RoutingStrategy(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterationCount = 0
        self.epochCount = 0

    def set_training_statistics(self, iteration_count, epoch_count):
        self.iterationCount = iteration_count
        self.epochCount = epoch_count

    def modify_temperature(self, softmax_decay_controller):
        pass

    def calculate_information_gain_losses(self, ig_losses, decision_loss_coefficient):
        pass

    def call(self, inputs, **kwargs):
        pass
