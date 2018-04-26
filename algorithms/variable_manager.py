import tensorflow as tf


class VariableManager:
    def __init__(self, network):
        self.network = network
        self.trainableVariables = []

    def save_trainable_variables(self):
        self.trainableVariables = tf.trainable_variables()

    def trainable_variables(self):
        return self.trainableVariables


