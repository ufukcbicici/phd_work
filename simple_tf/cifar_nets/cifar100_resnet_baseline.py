import tensorflow as tf
from simple_tf.global_params import GlobalConstants
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator


def baseline(node, network, variables=None):
    network.mask_input_nodes(node=node)
    # Input layer
    resnet_first_layer = ResnetGenerator.get_input(input=network.dataTensor, node=node,
                                                   resnet_hyperparams=GlobalConstants.RESNET_HYPERPARAMS)
    strides = GlobalConstants.RESNET_HYPERPARAMS.strides
    activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
    filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
    num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
    # Block 1
    with tf.variable_scope('block_1_0'):
        x = ResnetGenerator.residual(x=resnet_first_layer, in_filter=filters[0], out_filter=filters[1],
                                     stride=ResnetGenerator.stride_arr(strides[0]),
                                     activate_before_residual=activate_before_residual[0])
        for i in range(num_of_units_per_block):
            with tf.variable_scope("block_1_{0}".format(i + 1)):
                x = ResnetGenerator.residual(x=resnet_first_layer, in_filter=filters[1], out_filter=filters[1],
                                             stride=ResnetGenerator.stride_arr(1),
                                             activate_before_residual=False)
    # Block 2
    with tf.variable_scope('block_2_0'):
        x = ResnetGenerator.residual(x=resnet_first_layer, in_filter=filters[1], out_filter=filters[2],
                                     stride=ResnetGenerator.stride_arr(strides[1]),
                                     activate_before_residual=activate_before_residual[1])
        for i in range(num_of_units_per_block):
            with tf.variable_scope("block_2_{0}".format(i + 1)):
                x = ResnetGenerator.residual(x=resnet_first_layer, in_filter=filters[2], out_filter=filters[2],
                                             stride=ResnetGenerator.stride_arr(1),
                                             activate_before_residual=False)
    # Block 3
    with tf.variable_scope('block_3_0'):
        x = ResnetGenerator.residual(x=resnet_first_layer, in_filter=filters[2], out_filter=filters[3],
                                     stride=ResnetGenerator.stride_arr(strides[2]),
                                     activate_before_residual=activate_before_residual[2])
        for i in range(num_of_units_per_block):
            with tf.variable_scope("block_2_{0}".format(i + 1)):
                x = ResnetGenerator.residual(x=resnet_first_layer, in_filter=filters[3], out_filter=filters[3],
                                             stride=ResnetGenerator.stride_arr(1),
                                             activate_before_residual=False)
