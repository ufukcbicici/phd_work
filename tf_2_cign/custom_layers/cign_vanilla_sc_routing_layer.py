import tensorflow as tf


class CignVanillaScRoutingLayer(tf.keras.layers.Layer):
    def __init__(self, network):
        super().__init__()
        self.network = network
        # sself.level = level

    @tf.function
    def call(self, inputs, **kwargs):
        input_f_tensor = inputs[0]
        input_ig_routing_matrix = inputs[1]

        secondary_routing_matrix = tf.identity(input_ig_routing_matrix)
        return secondary_routing_matrix
