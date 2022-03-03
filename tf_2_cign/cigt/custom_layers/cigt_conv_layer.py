import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtConvLayer(CignConvLayer):
    def __init__(self, kernel_size, num_of_filters, strides, node, activation, use_bias=True, padding="same",
                 name="conv_op"):
        super().__init__(kernel_size, num_of_filters, strides, node, activation, use_bias, padding, name)
        self.cigtMaskingLayer = CigtMaskingLayer()

    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        net = self.layer(net)
        net = self.cigtMaskingLayer([net, routing_matrix])
        return net
