import numpy as np
import tensorflow as tf

from algorithms.info_gain import InfoGainLoss
from auxillary.dag_utilities import Dag
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_node import JungleNode, NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class Jungle_V2(FastTreeNetwork):
    def __init__(self, node_build_funcs, h_dimensions, dataset, network_name):
        super().__init__(node_build_funcs, None, None, None, None, None, dataset, network_name)
        curr_index = 0
        self.batchSize = tf.placeholder(name="batch_size", dtype=tf.int64)
        self.evalMultipath = tf.placeholder(name="eval_multipath", dtype=tf.int64)
        self.hDimensions = h_dimensions
        self.depthToNodesDict = {}
        self.currentGraph = tf.get_default_graph()
        self.batchIndices = tf.cast(tf.range(self.batchSize), dtype=tf.int32)
        self.nodeBuildFuncs = node_build_funcs
        self.nodes = {}

    def build_network(self):
        self.dagObject = Dag()
        self.nodes = {}
        curr_index = 0
        for depth, build_func in enumerate(self.nodeBuildFuncs):
            if depth == 0:
                node_type = NodeType.root_node
            elif depth == len(self.nodeBuildFuncs) - 1:
                node_type = NodeType.leaf_node
            else:
                node_type = NodeType.f_node
            curr_node = JungleNode(index=curr_index, depth=depth, node_type=node_type)
            self.nodes[curr_index] = curr_node
            node_output = build_func()
            if node_type != NodeType.leaf_node:
                self.information_gain_output(curr_node, node_output, self.hDimensions[depth])
            else:
                self.loss_output(node=curr_node, node_output=node_output)
            curr_index += 1

    def information_gain_output(self, node, node_output, h_dimension):
        assert len(node_output.get_shape().as_list()) == 2 and len(node_output.get_shape().as_list()) == 4
        if len(node_output.get_shape().as_list()) == 4:
            net_shape = node_output.get_shape().as_list()
            # Global Average Pooling
            h_net = tf.nn.avg_pool(net_shape, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1],
                                   padding='VALID')
            net_shape = h_net.get_shape().as_list()
            h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        else:
            h_net = node_output
        # Step 1: Create Hyperplanes
        ig_feature_size = node_output.get_shape().as_list()[-1]
        hyperplane_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
            shape=[ig_feature_size, h_dimension],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal([ig_feature_size, h_dimension], stddev=0.1,
                                            seed=GlobalConstants.SEED,
                                            dtype=GlobalConstants.DATA_TYPE))
        hyperplane_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
            shape=[h_dimension],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.0, shape=[h_dimension], dtype=GlobalConstants.DATA_TYPE))
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            h_net = tf.layers.batch_normalization(inputs=h_net,
                                                  momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                  training=tf.cast(self.isTrain, tf.bool))
        activations = FastTreeNetwork.fc_layer(x=h_net, W=hyperplane_weights, b=hyperplane_biases, node=node)
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
        p_F_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = self.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)

    def loss_output(self, node, node_output):
        output_feature_dim = node_output.get_shape().as_list()[-1]
        softmax_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="softmax_weights", node=node),
            shape=[output_feature_dim, self.labelCount],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal([output_feature_dim, self.labelCount], stddev=0.1,
                                            seed=GlobalConstants.SEED,
                                            dtype=GlobalConstants.DATA_TYPE))
        softmax_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="softmax_biases", node=node),
            shape=[self.labelCount],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.0, shape=[self.labelCount], dtype=GlobalConstants.DATA_TYPE))
        self.apply_loss(node=node, final_feature=node_output, softmax_weights=softmax_weights,
                        softmax_biases=softmax_biases)
