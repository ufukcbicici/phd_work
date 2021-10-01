import tensorflow as tf

from tf_2_cign.utilities.utilities import Utilities


class CignConvLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_of_filters, strides, node, activation,
                 use_bias=True, padding="same", name="conv_op"):
        super().__init__()
        if node is not None:
            op_id = 0
            while True:
                if "{0}_{1}".format(name, op_id) in node.opMacCostsDict:
                    op_id += 1
                    continue
                break
            op_name = "{0}_{1}".format(name, op_id)
        else:
            op_name = ""
        self.node = node
        self.opName = op_name
        self.kernelSize = kernel_size
        self.numOfFilters = num_of_filters
        self.strides = strides
        self.activation = activation
        self.useBias = use_bias
        self.padding = padding
        self.layer = tf.keras.layers.Conv2D(filters=num_of_filters,
                                            kernel_size=kernel_size,
                                            activation=activation,
                                            strides=strides,
                                            padding=padding,
                                            use_bias=use_bias,
                                            name=Utilities.get_variable_name(name="ConvLayer_{0}".format(op_name),
                                                                             node=node))

    def build(self, input_shape):
        assert len(input_shape.as_list()) == 4
        assert self.strides[0] == self.strides[1]
        # shape = [filter_size, filter_size, in_filters, out_filters]
        num_of_input_channels = input_shape.as_list()[3]
        height_of_input_map = input_shape.as_list()[2]
        width_of_input_map = input_shape.as_list()[1]
        height_of_filter = self.kernelSize
        width_of_filter = self.kernelSize
        num_of_output_channels = self.numOfFilters
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
        net = self.layer(inputs)
        return net
