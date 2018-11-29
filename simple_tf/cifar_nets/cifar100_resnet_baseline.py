import tensorflow as tf
from simple_tf.global_params import GlobalConstants
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator
from data_handling.cifar_dataset import CifarDataSet

def baseline(node, network, variables=None):
    network.mask_input_nodes(node=node)
    strides = GlobalConstants.RESNET_HYPERPARAMS.strides
    activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
    filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
    num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
    relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
    first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size

    # Input layer
    x = ResnetGenerator.get_input(input=network.dataTensor, node=node,
                                  first_conv_filter_size=first_conv_filter_size, out_filters=filters[0])
    # Block 1
    with tf.variable_scope('block_1_0'):
        x = ResnetGenerator.bottleneck_residual(x=x, node=node, in_filter=filters[0], out_filter=filters[1],
                                                stride=ResnetGenerator.stride_arr(strides[0]),
                                                activate_before_residual=activate_before_residual[0],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain)
        for i in range(num_of_units_per_block):
            with tf.variable_scope("block_1_{0}".format(i + 1)):
                x = ResnetGenerator.bottleneck_residual(x=x, node=node, in_filter=filters[1],
                                                        out_filter=filters[1],
                                                        stride=ResnetGenerator.stride_arr(1),
                                                        activate_before_residual=False,
                                                        relu_leakiness=relu_leakiness, is_train=network.isTrain)
    # Block 2
    with tf.variable_scope('block_2_0'):
        x = ResnetGenerator.bottleneck_residual(x=x, node=node, in_filter=filters[1], out_filter=filters[2],
                                                stride=ResnetGenerator.stride_arr(strides[1]),
                                                activate_before_residual=activate_before_residual[1],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain)
        for i in range(num_of_units_per_block):
            with tf.variable_scope("block_2_{0}".format(i + 1)):
                x = ResnetGenerator.bottleneck_residual(x=x, node=node, in_filter=filters[2],
                                                        out_filter=filters[2],
                                                        stride=ResnetGenerator.stride_arr(1),
                                                        activate_before_residual=False,
                                                        relu_leakiness=relu_leakiness, is_train=network.isTrain)
    # Block 3
    with tf.variable_scope('block_3_0'):
        x = ResnetGenerator.bottleneck_residual(x=x, node=node, in_filter=filters[2], out_filter=filters[3],
                                                stride=ResnetGenerator.stride_arr(strides[2]),
                                                activate_before_residual=activate_before_residual[2],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain)
        for i in range(num_of_units_per_block):
            with tf.variable_scope("block_2_{0}".format(i + 1)):
                x = ResnetGenerator.bottleneck_residual(x=x, node=node, in_filter=filters[3],
                                                        out_filter=filters[3],
                                                        stride=ResnetGenerator.stride_arr(1),
                                                        activate_before_residual=False,
                                                        relu_leakiness=relu_leakiness, is_train=network.isTrain)
    # Logit Layers
    with tf.variable_scope('unit_last'):
        x = ResnetGenerator.get_output(x=x, node=node, is_train=network.isTrain, leakiness=relu_leakiness)
    net_shape = x.get_shape().as_list()
    assert len(net_shape) == 4
    x = tf.reshape(x, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    output = x
    out_dim = network.dataset.get_label_count()
    weight = tf.get_variable(
        name=network.get_variable_name(name="fc_softmax_weights", node=node),
        shape=[x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    bias = tf.get_variable(network.get_variable_name(name="fc_softmax_biases", node=node), [out_dim],
                           initializer=tf.constant_initializer())
    # Loss
    final_feature, logits = network.apply_loss(node=node, final_feature=output,
                                               softmax_weights=weight, softmax_biases=bias)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor


def grad_func(network):
    pass


def threshold_calculator_func(network):
    pass


def residue_network_func(network):
    pass


def tensorboard_func(network):
    pass
