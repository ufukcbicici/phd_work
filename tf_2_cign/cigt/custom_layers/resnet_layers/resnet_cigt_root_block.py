import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_batch_norm_composite_layer import CigtConvBatchNormCompositeLayer
from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_identity_layer import CigtIdentityLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer
from tf_2_cign.cigt.custom_layers.inner_cigt_block import InnerCigtBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.basic_block import BasicBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_cigt_inner_block import ResnetCigtInnerBlock
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class ResnetCigtRootBlock(ResnetCigtInnerBlock):
    def __init__(self,
                 node,
                 first_conv_kernel_size,
                 first_conv_output_dim,
                 first_conv_stride,
                 block_parameters,
                 batch_norm_type,
                 start_moving_averages_from_zero,
                 apply_mask_to_batch_norm,
                 decision_drop_probability,
                 decision_average_pooling_stride,
                 decision_dim,
                 bn_momentum,
                 prev_block_path_count,
                 this_block_path_count,
                 next_block_path_count,
                 class_count,
                 ig_balance_coefficient,
                 routing_strategy,
                 use_straight_through,
                 decision_non_linearity):
        self.firstConvKernelSize = first_conv_kernel_size
        self.firstConvOutputDim = first_conv_output_dim
        self.firstConvStride = first_conv_stride
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

        super().__init__(node, block_parameters, batch_norm_type, start_moving_averages_from_zero,
                         apply_mask_to_batch_norm, decision_drop_probability, decision_average_pooling_stride,
                         decision_dim, bn_momentum, prev_block_path_count, this_block_path_count, next_block_path_count,
                         class_count, ig_balance_coefficient, routing_strategy, use_straight_through,
                         decision_non_linearity)

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        routing_matrix = inputs[1]
        temperature = inputs[2]
        labels = inputs[3]
        training = kwargs["training"]

        # We have an initial convolutional layer before the usual F layers.
        f_net = f_input
        f_net = self.firstConvBn([f_net, routing_matrix], training=training)
        f_net = tf.nn.relu(f_net)

        for block in self.blockList:
            f_net = block([f_net, routing_matrix], training=training)

        # H ops
        pre_branch_feature = f_net
        h_net = pre_branch_feature
        h_net = self.cigtRouteAveragingLayer([h_net, routing_matrix])
        h_net = self.decisionAveragePoolingLayer(h_net)
        h_net = self.decisionFlattenLayer(h_net)
        h_net = self.decisionFcLayer(h_net)

        # Decision layer
        ig_value, ig_activations_this, routing_probabilities, raw_activations, h_net_normed = \
            self.cigtDecisionLayer([h_net, labels, temperature], training=training)
        return f_net, ig_value, ig_activations_this, routing_probabilities, raw_activations, h_net_normed
