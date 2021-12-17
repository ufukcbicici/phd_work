import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


class CignTestLayer(tf.keras.layers.Layer):

    def __init__(self, level, network):
        super().__init__()
        self.level = level
        self.network = network

    def call(self, inputs, **kwargs):
        arr = inputs
        return tf.identity(inputs[0])
