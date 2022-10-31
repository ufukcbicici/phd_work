import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtIdentityLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        net = inputs
        x = tf.identity(net)
        return x
