import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


# OK
class CigjBlock(tf.keras.layers.Layer):

    def __init__(self, network):
        super().__init__()
        self.network = network

    def call(self, inputs, **kwargs):
        pass
