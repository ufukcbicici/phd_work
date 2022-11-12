import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_batch_norm_composite_layer import CigtConvBatchNormCompositeLayer
from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_identity_layer import CigtIdentityLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer
from tf_2_cign.cigt.custom_layers.inner_cigt_block import InnerCigtBlock
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class ResnetCigtInnerBlock(InnerCigtBlock):
    def __init__(self, node,
                 first_conv_kernel_size,
                 first_conv_output_dim,
                 first_conv_stride,
                 block_type,
                 block_parameters,
                 batch_norm_type,
                 start_moving_averages_from_zero,
                 apply_mask_to_batch_norm,
                 decision_drop_probability, decision_dim, bn_momentum,
                 prev_block_path_count,
                 this_block_path_count,
                 next_block_path_count,
                 class_count, ig_balance_coefficient, routing_strategy, use_straight_through, decision_non_linearity):
        super().__init__(node, routing_strategy, decision_drop_probability, decision_dim, bn_momentum,
                         prev_block_path_count,
                         this_block_path_count,
                         next_block_path_count,
                         class_count, ig_balance_coefficient,
                         use_straight_through, decision_non_linearity)
        self.firstConvKernelSize = first_conv_kernel_size
        self.firstConvOutputDim = first_conv_output_dim
        self.firstConvStride = first_conv_stride
        self.blockType = block_type
        self.blockParameters = block_parameters
        self.batchNormType = batch_norm_type
        self.startMovingAveragesFromZero = start_moving_averages_from_zero
        self.applyMaskToBatchNorm = apply_mask_to_batch_norm
        self.blockList = []

        # First convolutional layer + batch normalization
        if self.node.isRoot:
            self.firstConvBn = CigtConvBatchNormCompositeLayer(
                kernel_size=self.firstConvKernelSize,
                num_of_filters=self.firstConvOutputDim,
                strides=self.firstConvStride,
                node=node,
                activation=None,
                input_path_count=self.prevBlockPathCount,
                output_path_count=self.thisBlockPathCount,
                batch_norm_type=self.batchNormType,
                bn_momentum=self.bnMomentum,
                start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                use_bias=False,
                padding="same")
        else:
            self.firstConvBn = CigtIdentityLayer()

        # block_params_object is a dictionary with required parameters stored in key-value format.
        for block_params_object in self.blockParameters:
            output_dim = block_params_object["output_dim"]
            num_blocks = block_params_object["num_blocks"]
            stride = block_params_object["stride"]











#         self.convLayer = CigtConvLayer(kernel_size=kernel_size,
#                                        num_of_filters=num_of_filters,
#                                        strides=strides,
#                                        node=node,
#                                        activation=activation,
#                                        use_bias=use_bias,
#                                        padding=padding,
#                                        input_path_count=self.prevBlockPathCount,
#                                        output_path_count=self.thisBlockPathCount,
#                                        name="Lenet_Cigt_Node_{0}_Conv".format(self.node.index))
#         self.maxPoolLayer = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
#
#     def call(self, inputs, **kwargs):
#         f_input = inputs[0]
#         routing_matrix = inputs[1]
#         temperature = inputs[2]
#         labels = inputs[3]
#         training = kwargs["training"]
#
#         # F ops
#         f_net = self.convLayer([f_input, routing_matrix])
#         f_net = self.maxPoolLayer(f_net)
#
#         # H ops
#         pre_branch_feature = f_net
#         h_net = pre_branch_feature
#         h_net = self.cigtRouteAveragingLayer([h_net, routing_matrix])
#         h_net = self.decisionGAPLayer(h_net)
#         h_net = self.decisionFcLayer(h_net)
#         h_net = self.decisionDropoutLayer(h_net)
#
#         # Decision layer
#         ig_value, ig_activations_this, routing_probabilities, raw_activations, h_net_normed = \
#             self.cigtDecisionLayer([h_net, labels, temperature], training=training)
#         return f_net, ig_value, ig_activations_this, routing_probabilities, raw_activations, h_net_normed
