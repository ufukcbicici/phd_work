import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtDenseLayer(CignDenseLayer):
    def __init__(self, output_dim, activation, node, use_bias=True, name="fc_op"):
        super().__init__(output_dim, activation, node, use_bias, name)
        self.cigtMaskingLayer = CigtMaskingLayer()

    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        net = self.layer(net)
        net = self.cigtMaskingLayer([net, routing_matrix])
        return net
