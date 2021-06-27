import numpy as np
import tensorflow as tf

from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities import Utilities


class FashionNetLeafNodeFunc(tf.keras.layers.Layer):
    def __init__(self, node, network, kernel_size, num_of_filters, strides, activation, hidden_layer_dims,
                 classification_dropout_prob,
                 use_bias=True, padding="same"):
        super().__init__()
        self.node = node
        self.network = network
        # F Operations - Conv layer
        self.convLayer = CignConvLayer(kernel_size=kernel_size,
                                       num_of_filters=num_of_filters,
                                       strides=strides,
                                       node=node,
                                       activation=activation,
                                       use_bias=use_bias,
                                       padding=padding)
        self.maxPoolLayer = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        # F Operations - Dense Layers
        self.hiddenLayerDims = hidden_layer_dims
        self.flattenLayer = tf.keras.layers.Flatten()
        self.hiddenLayers = []
        self.dropoutLayers = []
        self.classificationDropoutProb = classification_dropout_prob
        for hidden_layer_dim in self.hiddenLayerDims:
            fc_layer = CignDenseLayer(output_dim=hidden_layer_dim,
                                      activation="relu",
                                      node=node,
                                      use_bias=True)
            self.hiddenLayers.append(fc_layer)
            dropout_layer = tf.keras.layers.Dropout(rate=self.classificationDropoutProb)
            self.dropoutLayers.append(dropout_layer)

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        h_input = inputs[1]
        ig_mask = inputs[2]
        sc_mask = inputs[3]

        # F ops -  # 1 Conv layer
        f_net = self.convLayer(f_input)
        f_net = self.maxPoolLayer(f_net)

        # F ops - Dense layers
        f_net = self.flattenLayer(f_net)
        for layer_id in range(len(self.hiddenLayers)):
            f_net = self.hiddenLayers[layer_id](f_net)
            f_net = self.dropoutLayers[layer_id](f_net)
        return f_net
