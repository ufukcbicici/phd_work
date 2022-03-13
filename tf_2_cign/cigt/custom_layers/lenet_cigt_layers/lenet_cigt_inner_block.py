import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class LeNetCigtInnerBlock(tf.keras.layers.Layer):
    def __init__(self, node, kernel_size, num_of_filters, strides, activation, use_bias, padding,
                 decision_drop_probability, decision_dim, bn_momentum,
                 next_block_path_count, class_count, ig_balance_coefficient):
        super().__init__()
        self.node = node
        self.bnMomentum = bn_momentum

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

        # H operations
        self.cigtRouteAveragingLayer = CigtRouteAveragingLayer()
        self.decisionGAPLayer = tf.keras.layers.GlobalAveragePooling2D()
        self.decisionDim = decision_dim
        self.decisionDropProbability = decision_drop_probability
        self.decisionFcLayer = CignDenseLayer(output_dim=self.decisionDim,
                                              activation="relu",
                                              node=node,
                                              use_bias=True,
                                              name="Lenet_Cigt_Node_{0}_fc_op_decision".format(self.node.index))
        self.decisionDropoutLayer = tf.keras.layers.Dropout(rate=self.decisionDropProbability)
        self.cigtDecisionLayer = CigtDecisionLayer(node=node,
                                                   decision_bn_momentum=self.bnMomentum,
                                                   next_block_path_count=next_block_path_count,
                                                   class_count=class_count,
                                                   ig_balance_coefficient=ig_balance_coefficient)

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
