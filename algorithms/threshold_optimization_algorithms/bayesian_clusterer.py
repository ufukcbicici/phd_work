import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite


class BayesianClusterer:
    def __init__(self, network, routing_data, cluster_count, fc_layers):
        with tf.variable_scope("clusterer"):
            root_node = [node for node in network.topologicalSortedNodes if node.isRoot]
            assert len(root_node) == 1
            root_node = root_node[0]
            feature_arr = routing_data.get_dict("pre_branch_feature")[root_node.index]
            entry_shape = list(feature_arr.shape)
            entry_shape[0] = None
            self.netInput = tf.placeholder(dtype=tf.float32, shape=entry_shape, name="net_input")
            net = self.netInput
            net = DeepQNetworks.global_average_pooling(net_input=net)
            hidden_layers = [fc_dim for fc_dim in fc_layers]
            hidden_layers.append(cluster_count)
            for layer_id, layer_dim in enumerate(hidden_layers):
                curr_dim = net.get_shape().as_list()[-1]
                W, b = FashionCignLite.get_affine_layer_params(
                    layer_shape=[curr_dim, layer_dim],
                    W_name="{0}_W_{1}".format("clusterer", layer_id),
                    b_name="{0}_b_{1}".format("clusterer", layer_id))
                net = FastTreeNetwork.fc_layer(x=net, W=W, b=b, node=root_node)
                if layer_id < len(hidden_layers) - 1:
                    net = tf.nn.relu(net)
                else:
                    self.clustererOutput = tf.nn.softmax(net)
