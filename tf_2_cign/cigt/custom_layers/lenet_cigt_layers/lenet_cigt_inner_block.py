import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer
from tf_2_cign.cigt.custom_layers.inner_cigt_block import InnerCigtBlock
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class LeNetCigtInnerBlock(InnerCigtBlock):
    def __init__(self, node, kernel_size, num_of_filters, strides, activation, use_bias, padding,
                 decision_drop_probability, decision_dim, bn_momentum, next_block_path_count, class_count,
                 ig_balance_coefficient, routing_strategy):

        super().__init__(node, routing_strategy, decision_drop_probability, decision_dim, bn_momentum,
                         next_block_path_count, class_count, ig_balance_coefficient)
        # F operations
        self.convLayer = CigtConvLayer(kernel_size=kernel_size,
                                       num_of_filters=num_of_filters,
                                       strides=strides,
                                       node=node,
                                       activation=activation,
                                       use_bias=use_bias,
                                       padding=padding,
                                       name="Lenet_Cigt_Node_{0}_Conv".format(self.node.index))
        self.maxPoolLayer = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        routing_matrix = inputs[1]
        temperature = inputs[2]
        labels = inputs[3]
        training = kwargs["training"]

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
        ig_value, ig_activations_this, routing_probabilities = \
            self.cigtDecisionLayer([h_net, labels, temperature], training=training)
        return f_net, ig_value, ig_activations_this, routing_probabilities
