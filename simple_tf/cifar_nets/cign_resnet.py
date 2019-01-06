import tensorflow as tf
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator

strides = GlobalConstants.RESNET_HYPERPARAMS.strides
activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size


def root_func(node, network):
    network.mask_input_nodes(node=node)
    # Input layer
    x = ResnetGenerator.get_input(input=network.dataTensor, out_filters=filters[0],
                                  first_conv_filter_size=first_conv_filter_size)
    # Block 1
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_1_0", node=node)):
        x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[0], out_filter=filters[1],
                                                stride=ResnetGenerator.stride_arr(strides[0]),
                                                activate_before_residual=activate_before_residual[0],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    for i in range(num_of_units_per_block-1):
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_1_{0}".format(i + 1), node=node)):
            x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[1],
                                                    out_filter=filters[1],
                                                    stride=ResnetGenerator.stride_arr(1),
                                                    activate_before_residual=False,
                                                    relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY)


def l1_func(node, network):
    network.mask_input_nodes(node=node)
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_2_0", node=node)):
        x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[1], out_filter=filters[2],
                                                stride=ResnetGenerator.stride_arr(strides[1]),
                                                activate_before_residual=activate_before_residual[1],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    for i in range(num_of_units_per_block-1):
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_2_{0}".format(i + 1), node=node)):
            x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[2],
                                                    out_filter=filters[2],
                                                    stride=ResnetGenerator.stride_arr(1),
                                                    activate_before_residual=False,
                                                    relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY)


def leaf_func(node, network):
    network.mask_input_nodes(node=node)
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_3_0", node=node)):
        x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[2], out_filter=filters[3],
                                                stride=ResnetGenerator.stride_arr(strides[2]),
                                                activate_before_residual=activate_before_residual[2],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    for i in range(num_of_units_per_block-1):
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_3_{0}".format(i + 1), node=node)):
            x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[3],
                                                    out_filter=filters[3],
                                                    stride=ResnetGenerator.stride_arr(1),
                                                    activate_before_residual=False,
                                                    relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    # Logit Layers
    with tf.variable_scope('unit_last'):
        x = ResnetGenerator.get_output(x=x, is_train=network.isTrain, leakiness=relu_leakiness,
                                       bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
