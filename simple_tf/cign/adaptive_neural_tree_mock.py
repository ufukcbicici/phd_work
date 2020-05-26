import numpy as np
import tensorflow as tf

from simple_tf.uncategorized.node import Node
from auxillary.dag_utilities import Dag
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants

ROUTER_HIDDEN_DIM = 16


def transformer_build_func(node, net):
    input_feature_map_count = net.get_shape().as_list()[-1]
    # output_feature_map_count = params[1]
    # filter_size = params[2]
    # use_pooling = params[3]
    net = FashionNetCigj.build_conv_layer(input=net, node=node, filter_size=5,
                                          num_of_input_channels=input_feature_map_count,
                                          num_of_output_channels=40,
                                          use_pooling=True, name_suffix="{0}".format(node.index))
    return net


def router_build_func(node, net):
    input_feature_map_count = net.get_shape().as_list()[-1]
    # output_feature_map_count = params[1]
    # filter_size = params[2]
    # use_pooling = params[3]
    net = FashionNetCigj.build_conv_layer(input=net, node=node, filter_size=5,
                                          num_of_input_channels=input_feature_map_count,
                                          num_of_output_channels=40,
                                          use_pooling=True, name_suffix="{0}".format(node.index))
    # Global Average Pooling
    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    # FC Layers
    input_dim = net.get_shape().as_list()[-1]
    net = FashionNetCigj.build_fc_layer(input=net, node=node, input_dim=input_dim, output_dim=ROUTER_HIDDEN_DIM,
                                        dropout_prob_tensor=1.0, name_suffix="fc_1_{0}".format(node.index))
    input_dim = net.get_shape().as_list()[-1]
    net = FashionNetCigj.build_fc_layer(input=net, node=node, input_dim=input_dim, output_dim=2,
                                        dropout_prob_tensor=1.0, name_suffix="fc_2_{0}".format(node.index))
    return net


def leaf_build_func(node, net):
    # Global Average Pooling
    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    input_dim = net.get_shape().as_list()[-1]
    net = FashionNetCigj.build_fc_layer(input=net, node=node, input_dim=input_dim, output_dim=10,
                                        dropout_prob_tensor=1.0, name_suffix="leaf_{0}".format(node.index))
    return net


def mock_mnist_adaptive_neural_tree():
    net = tf.placeholder(GlobalConstants.DATA_TYPE, shape=(None, 28, 28, 1), name="dataTensor")
    dag_object = Dag()
    # Node 0
    curr_index = 0
    node_0 = Node(index=curr_index, depth=0, is_root=True, is_leaf=False)
    dag_object.add_node(node=node_0)
    net = transformer_build_func(node_0, net)
    # Node 1
    curr_index = 1
    node_1 = Node(index=curr_index, depth=1, is_root=False, is_leaf=False)
    dag_object.add_edge(parent=node_0, child=node_1)
    net_x = transformer_build_func(node_1, net)
    # Node 2
    curr_index = 2
    node_2 = Node(index=curr_index, depth=2, is_root=False, is_leaf=False)
    dag_object.add_edge(parent=node_1, child=node_2)
    router_1 = router_build_func(node_2, net_x)
    # Node 3
    curr_index = 3
    node_3 = Node(index=curr_index, depth=3, is_root=False, is_leaf=True)
    dag_object.add_edge(parent=node_2, child=node_3)
    leaf_1 = leaf_build_func(node_3, net_x)
    # Node 4
    curr_index = 4
    node_4 = Node(index=curr_index, depth=3, is_root=False, is_leaf=False)
    dag_object.add_edge(parent=node_2, child=node_4)
    net = router_build_func(node_4, net_x)
    # Node 5
    curr_index = 5
    node_5 = Node(index=curr_index, depth=4, is_root=False, is_leaf=True)
    dag_object.add_edge(parent=node_4, child=node_5)
    leaf_2 = leaf_build_func(node_5, net_x)
    # Node 6
    curr_index = 6
    node_6 = Node(index=curr_index, depth=4, is_root=False, is_leaf=True)
    dag_object.add_edge(parent=node_4, child=node_6)
    leaf_3 = leaf_build_func(node_6, net_x)

    all_nodes = dag_object.get_topological_sort()
    total_mac_cost = 0.0
    for node in all_nodes:
        total_mac_cost += node.macCost
    print("total_mac_cost={0}".format(total_mac_cost))
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())
    print("total_mac_cost={0}".format(total_param_count))


mock_mnist_adaptive_neural_tree()