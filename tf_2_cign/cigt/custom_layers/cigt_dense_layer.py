import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtDenseLayer(CignDenseLayer):
    def __init__(self, output_dim, activation,
                 input_path_count,
                 output_path_count,
                 node, use_bias=True, name="fc_op"):
        super().__init__(output_dim, activation, node, use_bias, name)
        self.cigtMaskingLayer = CigtMaskingLayer()
        self.inputPathCount = input_path_count
        self.outputPathCount = output_path_count

    # def build(self, input_shape):
    #     assert len(input_shape) == 2
    #     tensor_shape = input_shape[0]
    #     super(CigtDenseLayer, self).build(input_shape=tensor_shape)

    def build(self, input_shape):
        assert len(input_shape.as_list()) == 2
        num_of_input_channels = int(input_shape[1] / self.inputPathCount)
        num_of_output_channels = int(self.outputDim / self.outputPathCount)
        cost = Utilities.calculate_mac_of_computation(num_of_input_channels=num_of_input_channels,
                                                      height_of_input_map=1,
                                                      width_of_input_map=1,
                                                      height_of_filter=1,
                                                      width_of_filter=1,
                                                      num_of_output_channels=num_of_output_channels,
                                                      convolution_stride=1,
                                                      type="fc")
        if self.node is not None:
            self.node.macCost += cost
            self.node.opMacCostsDict[self.opName] = cost
        
    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        net = self.layer(net)
        net = self.cigtMaskingLayer([net, routing_matrix])
        return net
