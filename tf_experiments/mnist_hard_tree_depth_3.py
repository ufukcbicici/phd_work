import tensorflow as tf
import numpy as np
import itertools

from auxillary.constants import ChannelTypes, PoolingType, ActivationType, InitType, ProblemType, TreeType, \
    GlobalInputNames
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.tf_layer_factory import TfLayerFactory
from auxillary.train_program import TrainProgram
from data_handling.mnist_data_set import MnistDataSet
from framework.hard_trees.tree_network import TreeNetwork
from framework.network_channel import NetworkChannel
from optimizers.sgd_optimizer import SgdOptimizer

np_seed = 88
np.random.seed(np_seed)


conv_1_feature_map_count = 20
conv_2_feature_map_count = 25
fc_unit_count = 125


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
        TfLayerFactory.create_convolutional_layer(
            node=node, channel=f_channel, input_tensor=x_image,
            conv_filter_shape=[5, 5, 1, conv_1_feature_map_count],
            conv_stride_shape=[1, 1, 1, 1], pooling_shape=[1, 2, 2, 1], conv_padding="SAME",
            pooling_stride_shape=[1, 2, 2, 1], pooling_padding="SAME", init_type=InitType.custom,
            activation_type=ActivationType.relu, pooling_type=PoolingType.max, post_fix=ChannelTypes.f_operator.value)
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
        TfLayerFactory.create_convolutional_layer(
            node=node, channel=f_channel, input_tensor=conv_input,
            conv_filter_shape=[5, 5, conv_1_feature_map_count, conv_2_feature_map_count],
            conv_stride_shape=[1, 1, 1, 1], pooling_shape=[1, 2, 2, 1], conv_padding="SAME",
            pooling_stride_shape=[1, 2, 2, 1], pooling_padding="SAME", init_type=InitType.custom,
            activation_type=ActivationType.relu, pooling_type=PoolingType.max, post_fix=ChannelTypes.f_operator.value)
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.h_operator) as h_channel:
        h_channel.add_operation(op=h_input)


def leaf_func(network, node):
    parent_node = get_parent(network=network, node=node)
    conv_input = network.add_nodewise_input(producer_node=parent_node, producer_channel=ChannelTypes.f_operator,
                                            producer_channel_index=0, dest_node=node)
    # F channel
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.f_operator) as f_channel:
        flattened_conv = f_channel.add_operation(op=tf.reshape(conv_input, [-1, 7 * 7 * conv_2_feature_map_count]))
        TfLayerFactory.create_fc_layer(node=node, channel=f_channel, input_tensor=flattened_conv,
                                       fc_shape=[7 * 7 * conv_2_feature_map_count, fc_unit_count],
                                       init_type=InitType.custom, activation_type=ActivationType.relu,
                                       post_fix=ChannelTypes.f_operator.value)


def main():
    dataset = MnistDataSet(validation_sample_count=5000)
    lr_list = [0.005, 0.0075, 0.01, 0.0125, 0.015]
    lr_periods = [100000, 100000/2, 100000/3, 100000/4, 100000/5]
    list_of_lists = [lr_list , lr_periods]
    for idx in itertools.product(*list_of_lists):
        lr = idx[0]
        lr_period = idx[1]
        train_program_path = UtilityFuncs.get_absolute_path(script_file=__file__, relative_path="train_program.json")
        train_program = TrainProgram(program_file=train_program_path)
        train_program.set_train_program_element(element_name="lr_initial", keywords=",", skipwords="", value=lr)
        train_program.set_train_program_element(element_name="lr_update_interval",
                                                keywords=",", skipwords="", value=lr_period)
        cnn_lenet = TreeNetwork(dataset=dataset, parameter_file=None, tree_degree=2, tree_type=TreeType.hard,
                                problem_type=ProblemType.classification,
                                train_program=train_program,
                                explanation="1000 Epochs, {0} lr decay period, {1} initial lr".format(lr_period, lr),
                                list_of_node_builder_functions=[root_func, l1_func, leaf_func])
        optimizer = SgdOptimizer(network=cnn_lenet, use_biased_gradient_estimates=True)
        cnn_lenet.set_optimizer(optimizer=optimizer)
        cnn_lenet.build_network()
        cnn_lenet.init_session()
        cnn_lenet.train()
main()