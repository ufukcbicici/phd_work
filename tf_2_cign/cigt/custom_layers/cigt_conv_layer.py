import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.utilities.utilities import Utilities


class CigtConvLayer(CignConvLayer):
    def __init__(self, kernel_size, num_of_filters, strides, node, activation,
                 input_path_count,
                 output_path_count,
                 use_bias=True, padding="same",
                 name="conv_op"):
        super().__init__(kernel_size, num_of_filters, strides, node, activation, use_bias, padding, name)
        self.inputPathCount = input_path_count
        self.outputPathCount = output_path_count
        self.cigtMaskingLayer = CigtMaskingLayer()

    # def build(self, input_shape):
    #     assert len(input_shape) == 2
    #     tensor_shape = input_shape[0]
    #     super(CigtConvLayer, self).build(tensor_shape)

    def build(self, input_shape):
        assert len(input_shape) == 2
        tensor_shape = input_shape[0]
        assert len(tensor_shape.as_list()) == 4
        assert self.strides[0] == self.strides[1]
        # shape = [filter_size, filter_size, in_filters, out_filters]
        num_of_input_channels = int(tensor_shape.as_list()[3] / self.inputPathCount)
        height_of_input_map = tensor_shape.as_list()[2]
        width_of_input_map = tensor_shape.as_list()[1]
        height_of_filter = self.kernelSize
        width_of_filter = self.kernelSize
        num_of_output_channels = int(self.numOfFilters / self.outputPathCount)
        convolution_stride = self.strides[0]
        cost = Utilities.calculate_mac_of_computation(
            num_of_input_channels=num_of_input_channels,
            height_of_input_map=height_of_input_map, width_of_input_map=width_of_input_map,
            height_of_filter=height_of_filter, width_of_filter=width_of_filter,
            num_of_output_channels=num_of_output_channels, convolution_stride=convolution_stride
        )
        self.node.macCost += cost
        self.node.opMacCostsDict[self.opName] = cost

    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        net = self.layer(net)
        net = self.cigtMaskingLayer([net, routing_matrix])
        return net
