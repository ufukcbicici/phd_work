import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_identity_layer import CigtIdentityLayer
from tf_2_cign.cigt.custom_layers.inner_cigt_block import InnerCigtBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.basic_block import BasicBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_input_transformation_layer import ResnetInputTransformationLayer


class ResnetCigtInnerBlock(InnerCigtBlock):
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
                 routing_strategy_name,
                 use_straight_through,
                 decision_non_linearity):
        super().__init__(node, routing_strategy_name, decision_drop_probability, decision_dim, bn_momentum,
                         prev_block_path_count,
                         this_block_path_count,
                         next_block_path_count,
                         class_count, ig_balance_coefficient,
                         use_straight_through, decision_non_linearity)
        self.firstConvKernelSize = first_conv_kernel_size
        self.firstConvOutputDim = first_conv_output_dim
        self.firstConvStride = first_conv_stride
        if self.firstConvKernelSize > 0 and \
                self.firstConvOutputDim > 0 and \
                self.firstConvStride > 0:
            self.inputTransformationLayer = ResnetInputTransformationLayer(
                node=node,
                prev_block_path_count=prev_block_path_count,
                this_block_path_count=this_block_path_count,
                batch_norm_type=batch_norm_type,
                bn_momentum=bn_momentum,
                start_moving_averages_from_zero=start_moving_averages_from_zero,
                apply_mask_to_batch_norm=apply_mask_to_batch_norm,
                first_conv_kernel_size=self.firstConvKernelSize,
                first_conv_output_dim=self.firstConvOutputDim,
                first_conv_stride=self.firstConvStride)
        else:
            self.inputTransformationLayer = CigtIdentityLayer()

        self.blockParameters = block_parameters
        self.batchNormType = batch_norm_type
        self.startMovingAveragesFromZero = start_moving_averages_from_zero
        self.applyMaskToBatchNorm = apply_mask_to_batch_norm
        self.decisionAveragePoolingStride = decision_average_pooling_stride
        self.blockList = []

        # block_params_object is a dictionary with required parameters stored in key-value format.
        for block_params_object in self.blockParameters:
            # Number of feature maps entering the block
            in_dimension = block_params_object["in_dimension"]
            # Number of feature maps exiting the block
            out_dimension = block_params_object["out_dimension"]
            # Number of routes entering the block
            input_path_count = block_params_object["input_path_count"]
            # Number of routes exiting the block
            output_path_count = block_params_object["output_path_count"]
            # Stride of the block's input convolution layer. When this is larger than 1, it means that we are going to
            # apply dimension reduction to feature maps.
            stride = block_params_object["stride"]

            block = BasicBlock(in_dimension=in_dimension,
                               out_dimension=out_dimension,
                               node=self.node,
                               input_path_count=input_path_count,
                               output_path_count=output_path_count,
                               batch_norm_type=self.batchNormType,
                               bn_momentum=self.bnMomentum,
                               start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                               apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                               stride=stride)
            self.blockList.append(block)

        #  Change the GAP Layer with average pooling with size
        self.decisionAveragePoolingLayer = tf.keras.layers.AveragePooling2D(
            pool_size=(self.decisionAveragePoolingStride, self.decisionAveragePoolingStride),
            strides=(self.decisionAveragePoolingStride, self.decisionAveragePoolingStride))
        self.decisionFlattenLayer = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        routing_matrix = inputs[1]
        temperature = inputs[2]
        labels = inputs[3]
        training = kwargs["training"]

        # F ops
        f_net = f_input
        # Input transformation
        f_net = self.inputTransformationLayer([f_net, routing_matrix], training=training)
        # Basic blocks
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
