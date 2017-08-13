import tensorflow as tf
import numpy as np
import itertools

from auxillary.constants import ChannelTypes, PoolingType, ActivationType, InitType, ProblemType, TreeType, \
    GlobalInputNames, parameterTypes
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.tf_layer_factory import TfLayerFactory
from auxillary.train_program import TrainProgram
from data_handling.mnist_data_set import MnistDataSet
from framework.hard_trees.tree_network import TreeNetwork
from framework.network_channel import NetworkChannel
from optimizers.sgd_optimizer import SgdOptimizer

np_seed = 88
np.random.seed(np_seed)

conv_1_feature_map_count = 32
conv_2_feature_map_count = 32
fc_unit_count = 256


def get_parent(network, node):
    parents = network.dag.parents(node=node)
    if len(parents) != 1:
        raise Exception("Invalid tree.")
    parent_node = parents[0]
    return parent_node


def root_func(network, node):
    # Data Input
    x = network.add_nodewise_input(producer_channel=ChannelTypes.data_input, dest_node=node)
    # Label Input
    y = network.add_nodewise_input(producer_channel=ChannelTypes.label_input, dest_node=node)
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.f_operator) as f_channel:
        # Reshape x for convolutions
        x_image = tf.reshape(x, [-1, MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1])
        # Convolution Filter 1
        conv_filter_shape = [5, 5, 1, conv_1_feature_map_count]
        conv_stride_shape = [1, 1, 1, 1]
        pooling_shape = [1, 2, 2, 1]
        conv_padding = "SAME"
        pooling_stride_shape = [1, 2, 2, 1]
        pooling_padding = "SAME"
        post_fix = ChannelTypes.f_operator.value
        W = node.create_variable(name="Convolution_Filter_{0}".format(post_fix),
                                 initializer=tf.truncated_normal(conv_filter_shape, stddev=0.1),
                                 dtype=tf.float32, arg_type=parameterTypes.learnable_parameter,
                                 channel=f_channel)
        b = node.create_variable(name="Convolution_Bias_{0}".format(post_fix),
                                 initializer=tf.constant(0.1, shape=[conv_filter_shape[3]]),
                                 dtype=tf.float32, arg_type=parameterTypes.learnable_parameter,
                                 channel=f_channel)
        # Operations
        conv_intermediate = f_channel.add_operation(op=tf.nn.conv2d(x_image, W,
                                                                    strides=conv_stride_shape, padding=conv_padding))
        conv = f_channel.add_operation(op=conv_intermediate + b)
        # Nonlinearity
        nonlinear_conv = f_channel.add_operation(op=tf.nn.relu(conv))
        # Pooling
        f_channel.add_operation(op=tf.nn.max_pool(nonlinear_conv, ksize=pooling_shape,
                                                  strides=pooling_stride_shape,
                                                  padding=pooling_padding))
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.h_operator) as h_channel:
        x_flattened = tf.reshape(x, [-1, MnistDataSet.MNIST_SIZE * MnistDataSet.MNIST_SIZE])
        h_channel.add_operation(op=x_flattened)


def l1_func(network, node):
    parent_node = get_parent(network=network, node=node)
    conv_input = network.add_nodewise_input(producer_node=parent_node, producer_channel=ChannelTypes.f_operator,
                                            producer_channel_index=0, dest_node=node)
    h_input = network.add_nodewise_input(producer_node=parent_node, producer_channel=ChannelTypes.h_operator,
                                         producer_channel_index=0, dest_node=node)
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.f_operator) as f_channel:
        # Convolution Filter 2
        conv_filter_shape = [5, 5, conv_1_feature_map_count, conv_2_feature_map_count]
        conv_stride_shape = [1, 1, 1, 1]
        pooling_shape = [1, 2, 2, 1]
        conv_padding = "SAME"
        pooling_stride_shape = [1, 2, 2, 1]
        pooling_padding = "SAME"
        post_fix = ChannelTypes.f_operator.value
        W = node.create_variable(name="Convolution_Filter_{0}".format(post_fix),
                                 initializer=tf.truncated_normal(conv_filter_shape, stddev=0.1),
                                 dtype=tf.float32, arg_type=parameterTypes.learnable_parameter,
                                 channel=f_channel)
        b = node.create_variable(name="Convolution_Bias_{0}".format(post_fix),
                                 initializer=tf.constant(0.1, shape=[conv_filter_shape[3]]),
                                 dtype=tf.float32, arg_type=parameterTypes.learnable_parameter,
                                 channel=f_channel)
        # Operations
        conv_intermediate = f_channel.add_operation(op=tf.nn.conv2d(conv_input, W,
                                                                    strides=conv_stride_shape, padding=conv_padding))
        conv = f_channel.add_operation(op=conv_intermediate + b)
        # Nonlinearity
        nonlinear_conv = f_channel.add_operation(op=tf.nn.relu(conv))
        # Pooling
        f_channel.add_operation(op=tf.nn.max_pool(nonlinear_conv, ksize=pooling_shape,
                                                  strides=pooling_stride_shape,
                                                  padding=pooling_padding))
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.h_operator) as h_channel:
        h_channel.add_operation(op=h_input)


def leaf_func(network, node):
    parent_node = get_parent(network=network, node=node)
    conv_input = network.add_nodewise_input(producer_node=parent_node, producer_channel=ChannelTypes.f_operator,
                                            producer_channel_index=0, dest_node=node)
    post_fix = ChannelTypes.f_operator.value
    fc_shape = [7 * 7 * conv_2_feature_map_count, fc_unit_count]
    # F channel
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.f_operator) as f_channel:
        flattened_conv = f_channel.add_operation(op=tf.reshape(conv_input, [-1, 7 * 7 * conv_2_feature_map_count]))
        W = node.create_variable(name="FullyConnected_Weight_{0}".format(post_fix),
                                 initializer=tf.truncated_normal(fc_shape, stddev=0.1),
                                 dtype=tf.float32, arg_type=parameterTypes.learnable_parameter, channel=f_channel)
        b = node.create_variable(name="FullyConnected_Bias_{0}".format(post_fix),
                                 initializer=tf.constant(0.1, shape=[fc_shape[1]]),
                                 dtype=tf.float32, arg_type=parameterTypes.learnable_parameter, channel=f_channel)
        # Operations
        matmul = f_channel.add_operation(op=tf.matmul(flattened_conv, W))
        fc = f_channel.add_operation(op=matmul + b)
        # Nonlinearity
        f_channel.add_operation(op=tf.nn.relu(fc))


def activation_generator_func(node, channel, network, feature_dimension, input_tensor):
    fc_shape = [feature_dimension, network.treeDegree]
    activation_tensor = TfLayerFactory.create_fc_layer(node=node, channel=channel,
                                                       input_tensor=input_tensor,
                                                       fc_shape=[feature_dimension, network.treeDegree],
                                                       weight_init=tf.truncated_normal(shape=fc_shape, stddev=0.1),
                                                       bias_init=tf.constant(0.1, shape=[fc_shape[1]]),
                                                       activation_type=ActivationType.no_activation,
                                                       post_fix=ChannelTypes.branching_activation.value)
    return activation_tensor


def loss_layer_generator_func(node, channel, feature_dimension, target_count, input_tensor):
    fc_shape = [feature_dimension, target_count]
    logitTensor = TfLayerFactory.create_fc_layer(node=node, channel=channel,
                                                 input_tensor=input_tensor,
                                                 fc_shape=fc_shape,
                                                 weight_init=tf.truncated_normal(shape=fc_shape, stddev=0.1),
                                                 bias_init=tf.constant(0.1, shape=[fc_shape[1]]),
                                                 activation_type=ActivationType.relu,
                                                 post_fix=ChannelTypes.pre_loss.value)
    return logitTensor


def main():
    dataset = MnistDataSet(validation_sample_count=10000)
    # lr_list = [0.01]
    # lr_periods = [1000]
    # list_of_lists = [lr_list , lr_periods]
    # for idx in itertools.product(*list_of_lists):
    #     lr = idx[0]
    #     lr_period = idx[1]
    #     train_program_path = UtilityFuncs.get_absolute_path(script_file=__file__, relative_path="train_program.json")
    #     train_program = TrainProgram(program_file=train_program_path)
    #     train_program.set_train_program_element(element_name="lr_initial", keywords=",", skipwords="", value=lr)
    #     train_program.set_train_program_element(element_name="lr_update_interval",
    #                                             keywords=",", skipwords="", value=lr_period)
    #     cnn_lenet = TreeNetwork(dataset=dataset, parameter_file=None, tree_degree=2, tree_type=TreeType.hard,
    #                             problem_type=ProblemType.classification,
    #                             train_program=train_program,
    #                             explanation="1000 Epochs, {0} lr decay period, {1} initial lr".format(lr_period, lr),
    #                             list_of_node_builder_functions=[root_func, l1_func, leaf_func])
    #     optimizer = SgdOptimizer(network=cnn_lenet, use_biased_gradient_estimates=True)
    #     cnn_lenet.set_optimizer(optimizer=optimizer)
    #     cnn_lenet.build_network()
    #     cnn_lenet.init_session()
    #     cnn_lenet.train()
    lr_list = [0.009]
    list_of_lists = [lr_list]
    for idx in itertools.product(*list_of_lists):
        tf.reset_default_graph()
        lr = idx[0]
        train_program_path = UtilityFuncs.get_absolute_path(script_file=__file__, relative_path="train_program.json")
        train_program = TrainProgram(program_file=train_program_path)
        train_program.set_train_program_element(element_name="lr_initial", keywords=",",
                                                skipwords={"BranchingActivation"},
                                                value=lr)
        # train_program.set_train_program_element(element_name="lr_update_interval",
        #                                         keywords=",", skipwords="", value=lr_period)
        cnn_lenet = TreeNetwork(dataset=dataset, parameter_file=None, tree_degree=2, tree_type=TreeType.hard,
                                problem_type=ProblemType.classification,
                                train_program=train_program,
                                # explanation="Saving Good Inits",
                                explanation="Branching -  Threshold fixed to 0.5 - lr fixed to {0}"
                                            "- Decisions are fixed - Biased Gradients - truncated normal init - "
                                            "relu activations".format(lr),
                                list_of_node_builder_functions=[root_func, l1_func, leaf_func],
                                activation_generator_func=activation_generator_func,
                                loss_layer_generator_func=loss_layer_generator_func)
        optimizer = SgdOptimizer(network=cnn_lenet, use_biased_gradient_estimates=True)
        cnn_lenet.set_optimizer(optimizer=optimizer)
        cnn_lenet.build_network()
        cnn_lenet.init_session()
        cnn_lenet.train()


main()
