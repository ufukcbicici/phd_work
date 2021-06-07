import numpy as np
import tensorflow as tf

from tf_2_cign.utilities import Utilities


class CignDenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation, node, use_bias=True, name="fc_op"):
        super().__init__()
        if node is not None:
            # node.macCost += cost
            op_id = 0
            while True:
                if "{0}_{1}".format(name, op_id) in node.opMacCostsDict:
                    op_id += 1
                    continue
                break
            op_name = "{0}_{1}".format(name, op_id)
            # node.opMacCostsDict[op_name] = cost
        else:
            op_name = ""
        self.node = node
        self.opName = op_name
        self.outputDim = output_dim
        if self.node is not None:
            self.layer = tf.keras.layers.Dense(
                units=self.outputDim,
                activation=activation,
                use_bias=use_bias,
                name=Utilities.get_variable_name(name="DenseLayer_{0}".format(self.opName),
                                                 node=node))
        else:
            self.layer = tf.keras.layers.Dense(
                units=self.outputDim,
                activation=activation,
                use_bias=use_bias,
                name="DenseLayer")

    def build(self, input_shape):
        assert len(input_shape.as_list()) == 2
        num_of_input_channels = input_shape[1]
        num_of_output_channels = self.outputDim
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
        net = self.layer(inputs)
        return net
