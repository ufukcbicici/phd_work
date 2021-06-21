import numpy as np
import tensorflow as tf

from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities import Utilities


class FashionNetInnerNodeFunc(tf.keras.layers.Layer):
    def __init__(self, node, network, kernel_size, num_of_filters, strides, activation, decision_dim,
                 decision_drop_probability,
                 use_bias=True, padding="same"):
        super().__init__()
        self.node = node
        self.network = network
        self.convLayer = CignConvLayer(kernel_size=kernel_size,
                                       num_of_filters=num_of_filters,
                                       strides=strides,
                                       node=node,
                                       activation=activation,
                                       use_bias=use_bias,
                                       padding=padding)
        self.decisionDim = decision_dim
        self.decisionDropProbability = decision_drop_probability
        self.decisionFcLayer = CignDenseLayer(output_dim=self.decisionDim,
                                              activation="relu",
                                              node=node,
                                              use_bias=True,
                                              name="fc_op_decision")
        self.decisionGAPLayer = tf.keras.layers.GlobalAveragePooling2D()
        self.decisionDropoutLayer = tf.keras.layers.Dropout(rate=self.decisionDropProbability)

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        h_input = inputs[1]
        ig_mask = inputs[2]
        sc_mask = inputs[3]

        # F ops
        f_net = self.convLayer(f_input)

        # H Ops
        pre_branch_feature = f_net
        h_net = self.decisionGAPLayer(pre_branch_feature)
        h_net = self.decisionFcLayer(h_net)
        h_net = self.decisionDropoutLayer(h_net)

        return f_net, h_net, pre_branch_feature
