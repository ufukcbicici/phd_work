import tensorflow as tf

from auxillary.constants import ChannelTypes, InitType, ActivationType, PoolingType, TreeType, \
    ProblemType
from auxillary.tf_layer_factory import TfLayerFactory
from data_handling.mnist_data_set import MnistDataSet
from framework.hard_trees.tree_network import TreeNetwork
from framework.network_channel import NetworkChannel


def baseline_network(network, node):
    # Data Input
    x = network.add_nodewise_input(producer_node=None, producer_channel=ChannelTypes.data_input, producer_channel_index=0,
                                   dest_node=node)
    # Label Input
    y = network.add_nodewise_input(producer_node=None, producer_channel=ChannelTypes.label_input, producer_channel_index=0,
                                   dest_node=node)
    # F channel
    with NetworkChannel(parent_node=node, parent_node_channel=ChannelTypes.f_operator) as f_channel:
        # Reshape x for convolutions
        x_image = tf.reshape(x, [-1, MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1])
        # Convolution Filter 1
        conv_1_feature_map_count = 20
        conv_layer_1 = TfLayerFactory.create_convolutional_layer(
            node=node, channel=f_channel, input_tensor=x_image,
            conv_filter_shape=[5, 5, 1, conv_1_feature_map_count],
            conv_stride_shape=[1, 1, 1, 1], pooling_shape=[1, 2, 2, 1], conv_padding="SAME",
            pooling_stride_shape=[1, 2, 2, 1], pooling_padding="SAME", init_type=InitType.xavier,
            activation_type=ActivationType.relu, pooling_type=PoolingType.max, post_fix="1")
        # Convolution Filter 2
        conv_2_feature_map_count = 50
        conv_layer_2 = TfLayerFactory.create_convolutional_layer(
            node=node, channel=f_channel, input_tensor=conv_layer_1,
            conv_filter_shape=[5, 5, conv_1_feature_map_count, conv_2_feature_map_count],
            conv_stride_shape=[1, 1, 1, 1], pooling_shape=[1, 2, 2, 1], conv_padding="SAME",
            pooling_stride_shape=[1, 2, 2, 1], pooling_padding="SAME", init_type=InitType.xavier,
            activation_type=ActivationType.relu, pooling_type=PoolingType.max, post_fix="2")
        # Fully Connected Layer 1
        fc_unit_count = 500
        flattened_conv = f_channel.add_operation(
            op=tf.reshape(conv_layer_2, [-1, 7 * 7 * conv_2_feature_map_count]))
        TfLayerFactory.create_fc_layer(node=node, channel=f_channel, input_tensor=flattened_conv,
                                       fc_shape=[7 * 7 * conv_2_feature_map_count, fc_unit_count],
                                       init_type=InitType.xavier, activation_type=ActivationType.relu,
                                       post_fix="3")


def main():
    dataset = MnistDataSet(validation_sample_count=5000)
    dataset.load_dataset()
    cnn_lenet = TreeNetwork(run_id=0, dataset=dataset, parameter_file=None, tree_degree=2, tree_type=TreeType.hard,
                            problem_type=ProblemType.classification,
                            list_of_node_builder_functions=[baseline_network])
    cnn_lenet.build_network()


# def test(scp):
#     x = tf.placeholder(tf.float32, name="A")
#     y = tf.placeholder(tf.float32, name="B")
#     z = tf.placeholder(tf.float32, name="C")
#     tf.add_to_collection(name="Den", value=x)
#     tf.add_to_collection(name="Den", value=y)
#     tf.add_to_collection(name="Den", value=z)

main()

# dataset = MnistDataSet(validation_sample_count=5000)
# dataset.load_dataset()
# total_samples_seen = 0
#
# W1_shape = [5, 5, 1, 32]
# b1_shape = [32]
#
# x = tf.placeholder(tf.float32)
# initial_W1 = tf.truncated_normal(shape=W1_shape, stddev=0.1)
# W1 = tf.Variable(tf.truncated_normal(shape=W1_shape, stddev=0.1))
# initial_b1 = tf.constant(0.1, shape=b1_shape)
# b1 = tf.Variable(initial_b1)
# conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
# conv1_sum = conv1 + b1
# # y = tf.placeholder(tf.float32)
# # z = conv1 + y
# # dif = z - conv1
#
#
# sess = tf.Session()
#
# # Run init ops
# init = tf.global_variables_initializer()
# sess.run(init)
#
# while True:
#     samples, labels, indices = dataset.get_next_batch(batch_size=1000)
#     samples = samples.reshape((1000, MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1))
#     conv1_sum_res, W1_res, initial_W1_res = sess.run([conv1_sum, W1, initial_W1], feed_dict={x: samples})
#     print(W1_res[0, 0, 0, 0])
#     print(initial_W1_res[0, 0, 0, 0])
#     if dataset.isNewEpoch:
#         break
#
# print("X")
