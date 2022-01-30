import tensorflow as tf

from tf_2_cign.cigj.custom_layers.cigj_block import CigjBlock
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer


class FashionCigjNetBlock(CigjBlock):
    def __init__(self, network, node,  ig_activations, routing_matrix,
                 kernel_size, num_of_filters, strides, activation, use_bias, padding,
                 decision_drop_probability, decision_dim):
        super().__init__(network, node, ig_activations, routing_matrix)
        # F operations
        self.convLayer = CignConvLayer(kernel_size=kernel_size,
                                       num_of_filters=num_of_filters,
                                       strides=strides,
                                       node=node,
                                       activation=activation,
                                       use_bias=use_bias,
                                       padding=padding)
        self.maxPoolLayer = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

        # H operations
        self.decisionGAPLayer = tf.keras.layers.GlobalAveragePooling2D()
        self.decisionDim = decision_dim
        self.decisionDropProbability = decision_drop_probability
        self.decisionFcLayer = CignDenseLayer(output_dim=self.decisionDim,
                                              activation="relu",
                                              node=node,
                                              use_bias=True,
                                              name="fc_op_decision")
        self.decisionDropoutLayer = tf.keras.layers.Dropout(rate=self.decisionDropProbability)

    @tf.function
    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        h_input = inputs[1]
        ig_mask = inputs[2]
        sc_mask = inputs[3]

        # F ops
        f_net = self.convLayer(f_input)
        f_net = self.maxPoolLayer(f_net)

        # H Ops
        pre_branch_feature = f_net
        h_net = self.decisionGAPLayer(pre_branch_feature)
        h_net = self.decisionFcLayer(h_net)
        h_net = self.decisionDropoutLayer(h_net)

        return f_net, h_net, pre_branch_feature
