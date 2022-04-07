import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.cigt.custom_layers.cigt_masking_layer_with_boolean_mask import CigtMaskingLayerWithBooleanMask
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtConvLayer(CignConvLayer):
    def __init__(self, use_boolean_mask_layer,
                 kernel_size, num_of_filters, strides, node, activation, use_bias=True, padding="same",
                 name="conv_op"):
        super().__init__(kernel_size, num_of_filters, strides, node, activation, use_bias, padding, name)
        if not use_boolean_mask_layer:
            self.cigtMaskingLayer = CigtMaskingLayer()
        else:
            self.cigtMaskingLayer = CigtMaskingLayerWithBooleanMask()

    def build(self, input_shape):
        assert len(input_shape) == 2
        tensor_shape = input_shape[0]
        super(CigtConvLayer, self).build(tensor_shape)

    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        net = self.layer(net)
        net = self.cigtMaskingLayer([net, routing_matrix])
        return net
