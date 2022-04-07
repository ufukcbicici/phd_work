import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.cigt.custom_layers.cigt_masking_layer_with_boolean_mask import CigtMaskingLayerWithBooleanMask
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtDenseLayer(CignDenseLayer):
    def __init__(self, use_boolean_mask_layer,
                 output_dim, activation, node, use_bias=True, name="fc_op"):
        super().__init__(output_dim, activation, node, use_bias, name)
        if not use_boolean_mask_layer:
            self.cigtMaskingLayer = CigtMaskingLayer()
        else:
            self.cigtMaskingLayer = CigtMaskingLayerWithBooleanMask()

    def build(self, input_shape):
        assert len(input_shape) == 2
        tensor_shape = input_shape[0]
        super(CigtDenseLayer, self).build(input_shape=tensor_shape)
        
    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        net = self.layer(net)
        net = self.cigtMaskingLayer([net, routing_matrix])
        return net
