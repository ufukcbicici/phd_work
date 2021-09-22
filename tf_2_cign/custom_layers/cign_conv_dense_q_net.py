import numpy as np
import tensorflow as tf

from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.cign_rl_routing_layer import CignRlRoutingLayer
from tf_2_cign.utilities import Utilities


class CignConvDenseQNet(tf.keras.layers.Layer):
    def __init__(self, level,
                 node, network, kernel_size, num_of_filters, strides, activation,
                 hidden_layer_dims,
                 q_network_dim,
                 rl_dropout_prob,
                 use_bias=True, padding="same"):
        super().__init__()
        with tf.name_scope("Node_{0}".format(node.index)):
            with tf.name_scope("q_net"):
                self.level = level
                self.node = node
                self.network = network
                self.actionSpaces = [tf.constant(self.network.actionSpaces[idx])
                                     for idx in range(len(self.network.actionSpaces))]
                # F Operations - Conv layer
                self.convLayer = CignConvLayer(kernel_size=kernel_size,
                                               num_of_filters=num_of_filters,
                                               strides=strides,
                                               node=node,
                                               activation=activation,
                                               use_bias=use_bias,
                                               padding=padding,
                                               name="q_net_conv_layer")
                self.globalAveragingPoolLayer = tf.keras.layers.GlobalAveragePooling2D()
                # F Operations - Dense Layers
                self.hiddenLayerDims = hidden_layer_dims
                self.flattenLayer = tf.keras.layers.Flatten()
                self.hiddenLayers = []
                self.dropoutLayers = []
                self.rlDropoutProb = rl_dropout_prob
                for layer_id, hidden_layer_dim in enumerate(self.hiddenLayerDims):
                    fc_layer = CignDenseLayer(output_dim=hidden_layer_dim,
                                              activation="relu",
                                              node=node,
                                              use_bias=True,
                                              name="q_net_fc_layer_{0}".format(layer_id))
                    self.hiddenLayers.append(fc_layer)
                    dropout_layer = tf.keras.layers.Dropout(rate=self.rlDropoutProb)
                    self.dropoutLayers.append(dropout_layer)
                self.qNetworkDim = q_network_dim
                self.qNetLayer = CignDenseLayer(output_dim=self.qNetworkDim,
                                                activation=None,
                                                node=node,
                                                use_bias=True,
                                                name="q_net_q_table_layer")

    # @tf.function
    def call(self, inputs, **kwargs):
        input_f_tensor = inputs
        # input_ig_routing_matrix = inputs[1]
        # is_warm_up_period = inputs[2]
        # past_actions = inputs[3]

        # Apply a single conv layer first.
        q_net = self.convLayer(input_f_tensor)
        # Then apply global average pooling
        q_net = self.globalAveragingPoolLayer(q_net)

        # Dense Layers
        for layer_id in range(len(self.hiddenLayers)):
            q_net = self.hiddenLayers[layer_id](q_net)
            q_net = self.dropoutLayers[layer_id](q_net)

        # Output q predictions
        q_table_predicted = self.qNetLayer(q_net)
        return q_table_predicted
