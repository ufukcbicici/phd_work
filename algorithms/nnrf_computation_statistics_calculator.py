import numpy as np
import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class NNRFComputationStatisticsCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate(feature_size, class_count):
        m = feature_size
        C = class_count
        r = int(np.asscalar(np.ceil(np.sqrt(m))))
        d = int(np.asscalar(np.floor(np.log(C) / np.log(2)) + 1))
        N = 150
        network = FastTreeNetwork.get_mock_tree(degree_list=d * [2], network_name="NNRF")
        # Build a hypothetic network
        node_inputs_dict = {}
        node_outputs_dict = {}
        for node in network.topologicalSortedNodes:
            input_dim = r if node.isRoot else r + 1
            node_input = tf.placeholder(dtype=tf.float32, shape=(None, input_dim),
                                        name="node_input_{0}".format(node.index))
            node_inputs_dict[node.index] = node_input
            W_node = UtilityFuncs.create_variable(
                name=network.get_variable_name(name="W", node=node),
                shape=[input_dim, 2],
                dtype=GlobalConstants.DATA_TYPE,
                initializer=tf.truncated_normal(
                    [input_dim, 2], stddev=0.1, seed=GlobalConstants.SEED,
                    dtype=GlobalConstants.DATA_TYPE))
            b_node = UtilityFuncs.create_variable(
                name=network.get_variable_name(name="b", node=node),
                shape=[2],
                dtype=GlobalConstants.DATA_TYPE,
                initializer=tf.constant(0.1, shape=[2], dtype=GlobalConstants.DATA_TYPE))
            node_output = FastTreeNetwork.fc_layer(x=node_input, W=W_node, b=b_node, node=node)
            node_outputs_dict[node.index] = node_output
            # Add last layer to root
            if node.isRoot:
                last_layer_input = tf.placeholder(dtype=tf.float32, shape=(None, 2**d), name="last_layer_input")
                node_inputs_dict[-1] = node_input
                W_C = UtilityFuncs.create_variable(
                    name=network.get_variable_name(name="W_C", node=node),
                    shape=[2**d, C],
                    dtype=GlobalConstants.DATA_TYPE,
                    initializer=tf.truncated_normal(
                        [2**d, C], stddev=0.1, seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE))
                b_C = UtilityFuncs.create_variable(
                    name=network.get_variable_name(name="b_C", node=node),
                    shape=[C],
                    dtype=GlobalConstants.DATA_TYPE,
                    initializer=tf.constant(0.1, shape=[C], dtype=GlobalConstants.DATA_TYPE))
                logits = FastTreeNetwork.fc_layer(x=last_layer_input, W=W_C, b=b_C, node=node)
                node_outputs_dict[-1] = logits
        root_to_leaf_path = []
        curr_node = network.topologicalSortedNodes[0]
        while True:
            root_to_leaf_path.append(curr_node)
            children = network.dagObject.children(node=curr_node)
            if len(children) == 0:
                break
            curr_node = children[0]
        tree_mac_cost = sum([sum(node.opMacCostsDict.values()) for node in root_to_leaf_path])
        forest_mac_cost = N * tree_mac_cost
        tree_param_count = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
        forest_param_count = N * tree_param_count
        print("forest_mac_cost={0} forest_param_count={1}".format(forest_mac_cost, forest_param_count))
