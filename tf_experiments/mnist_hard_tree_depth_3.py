import tensorflow as tf

from auxillary.constants import ChannelTypes, PoolingType, ActivationType, InitType, ProblemType, TreeType, \
    GlobalInputNames
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.tf_layer_factory import TfLayerFactory
from auxillary.train_program import TrainProgram
from data_handling.mnist_data_set import MnistDataSet
from framework.hard_trees.tree_network import TreeNetwork
from framework.network_channel import NetworkChannel
from optimizers.sgd_optimizer import SgdOptimizer

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
            pooling_stride_shape=[1, 2, 2, 1], pooling_padding="SAME", init_type=InitType.xavier,
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
            pooling_stride_shape=[1, 2, 2, 1], pooling_padding="SAME", init_type=InitType.xavier,
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
                                       init_type=InitType.xavier, activation_type=ActivationType.relu,
                                       post_fix=ChannelTypes.f_operator.value)


def main():
    dataset = MnistDataSet(validation_sample_count=5000)
    dataset.load_dataset()
    train_program_path = UtilityFuncs.get_absolute_path(script_file=__file__, relative_path="train_program.json")
    train_program = TrainProgram(program_file=train_program_path)
    cnn_lenet = TreeNetwork(run_id=0, dataset=dataset, parameter_file=None, tree_degree=2, tree_type=TreeType.hard,
                            problem_type=ProblemType.classification,
                            train_program=train_program,
                            list_of_node_builder_functions=[root_func, l1_func, leaf_func])
    optimizer = SgdOptimizer(network=cnn_lenet, use_biased_gradient_estimates=True)
    cnn_lenet.set_optimizer(optimizer=optimizer)
    cnn_lenet.build_network()
    cnn_lenet.init_session()
    cnn_lenet.train()
main()