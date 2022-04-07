import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_dense_layer import CigtDenseLayer
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer


class LeNetCigtLeafBlock(tf.keras.layers.Layer):
    def __init__(self, use_boolean_mask_layer, node, kernel_size, num_of_filters, strides, activation, hidden_layer_dims,
                 classification_dropout_prob, class_count,
                 use_bias=True, padding="same"):
        super().__init__()
        self.useBooleanMaskLayer = use_boolean_mask_layer
        self.node = node
        # F Operations - Conv layer
        if num_of_filters is not None:
            self.convLayer = CigtConvLayer(use_boolean_mask_layer=self.useBooleanMaskLayer,
                                           kernel_size=kernel_size,
                                           num_of_filters=num_of_filters,
                                           strides=strides,
                                           node=node,
                                           activation=activation,
                                           use_bias=use_bias,
                                           padding=padding,
                                           name="Lenet_Cigt_Node_{0}_Conv".format(self.node.index))
            self.maxPoolLayer = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        else:
            self.convLayer = None
            self.maxPoolLayer = None

        # F Operations - Dense Layers
        self.hiddenLayerDims = hidden_layer_dims
        self.flattenLayer = tf.keras.layers.Flatten()
        self.hiddenLayers = []
        self.dropoutLayers = []
        self.classificationDropoutProb = classification_dropout_prob
        for hidden_layer_dim in self.hiddenLayerDims:
            fc_layer = CigtDenseLayer(use_boolean_mask_layer=self.useBooleanMaskLayer,
                                      output_dim=hidden_layer_dim,
                                      activation="relu",
                                      node=node,
                                      use_bias=True,
                                      name="Lenet_Cigt_Node_{0}_fc".format(self.node.index))
            self.hiddenLayers.append(fc_layer)
            dropout_layer = tf.keras.layers.Dropout(rate=self.classificationDropoutProb)
            self.dropoutLayers.append(dropout_layer)
        self.lossLayer = CignDenseLayer(output_dim=class_count, activation=None, node=node, use_bias=True,
                                        name="loss_layer")

    # @tf.function
    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        # ig_activations_parent = inputs[1]
        routing_matrix = inputs[1]
        labels = tf.cast(inputs[2], dtype=tf.int32)
        training = kwargs["training"]

        # F ops -  # 1 Conv layer
        if self.convLayer is not None and self.maxPoolLayer is not None:
            f_net = self.convLayer([f_input, routing_matrix])
            f_net = self.maxPoolLayer(f_net)
        else:
            f_net = tf.identity(f_input)

        # F ops - Dense layers
        f_net = self.flattenLayer(f_net)
        for layer_id in range(len(self.hiddenLayers)):
            f_net = self.hiddenLayers[layer_id]([f_net, routing_matrix])
            f_net = self.dropoutLayers[layer_id](f_net)

        # Loss layer
        logits = self.lossLayer(f_net)
        posteriors = tf.nn.softmax(logits)
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        classification_loss = tf.reduce_mean(cross_entropy_loss_tensor)

        return logits, posteriors, classification_loss

































