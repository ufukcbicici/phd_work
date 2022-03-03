import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class FashionCigtNetInnerBlock(tf.keras.layers.Layer):
    def __init__(self, network, node, kernel_size, num_of_filters, strides, activation, use_bias, padding,
                 decision_drop_probability, decision_dim, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.node = node

        # F operations
        self.convLayer = CigtConvLayer(kernel_size=kernel_size,
                                       num_of_filters=num_of_filters,
                                       strides=strides,
                                       node=node,
                                       activation=activation,
                                       use_bias=use_bias,
                                       padding=padding)
        self.maxPoolLayer = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

        # H operations
        self.cigtRouteAveragingLayer = CigtRouteAveragingLayer()
        self.decisionGAPLayer = tf.keras.layers.GlobalAveragePooling2D()
        self.decisionDim = decision_dim
        self.decisionDropProbability = decision_drop_probability
        self.decisionFcLayer = CignDenseLayer(output_dim=self.decisionDim,
                                              activation="relu",
                                              node=node,
                                              use_bias=True,
                                              name="fc_op_decision")
        self.cigtDecisionLayer = CigtDecisionLayer(network=self, node=node, decision_bn_momentum=self.bnMomentum)

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        ig_activations_parent = inputs[1]
        routing_matrix = inputs[2]
        temperature = inputs[3]
        labels = inputs[4]

        # F ops
        f_net = self.convLayer([f_input, routing_matrix])
        f_net = self.maxPoolLayer(f_net)

        # H ops
        pre_branch_feature = f_net
        h_net = pre_branch_feature
        h_net = self.cigtRouteAveragingLayer([h_net, routing_matrix])
        h_net = self.decisionGAPLayer(h_net)
        h_net = self.decisionFcLayer(h_net)
        h_net = self.decisionDropoutLayer(h_net)

        # Decision layer
        ig_value, ig_activations_this = self.cigtDecisionLayer([h_net, labels, temperature])
        return f_net, ig_value, ig_activations_this
