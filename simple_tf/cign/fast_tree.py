from collections import deque

import numpy as np
import tensorflow as tf
import time
import os
import pickle

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import RoutingDataset
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.tree import TreeNetwork
from simple_tf.global_params import GlobalConstants, AccuracyCalcType, TrainingUpdateResult
from algorithms.info_gain import InfoGainLoss
from simple_tf.uncategorized.node import Node
from auxillary.constants import DatasetTypes
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.framework.python.framework import checkpoint_utils


class FastTreeNetwork(TreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset)
        self.learningRate = None
        self.optimizer = None
        self.infoGainDicts = None
        self.extra_update_ops = None
        self.orderedNodesPerLevel = {}
        self.nodeCosts = {}
        self.opCosts = {}
        self.networkName = network_name
        self.dbName = None
        self.saver = None
        self.batchSizeTf = tf.placeholder(dtype=tf.int32, name="batch_size")

    @staticmethod
    def conv_layer(x, kernel, strides, node, bias=None, padding='SAME', name="conv_op"):
        assert len(x.get_shape().as_list()) == 4
        assert len(kernel.get_shape().as_list()) == 4
        assert kernel.get_shape().as_list()[2] == x.get_shape().as_list()[3]
        assert strides[1] == strides[2]
        # shape = [filter_size, filter_size, in_filters, out_filters]
        num_of_input_channels = x.get_shape().as_list()[3]
        height_of_input_map = x.get_shape().as_list()[2]
        width_of_input_map = x.get_shape().as_list()[1]
        height_of_filter = kernel.get_shape().as_list()[1]
        width_of_filter = kernel.get_shape().as_list()[0]
        num_of_output_channels = kernel.get_shape().as_list()[3]
        convolution_stride = strides[1]
        cost = UtilityFuncs.calculate_mac_of_computation(
            num_of_input_channels=num_of_input_channels,
            height_of_input_map=height_of_input_map, width_of_input_map=width_of_input_map,
            height_of_filter=height_of_filter, width_of_filter=width_of_filter,
            num_of_output_channels=num_of_output_channels, convolution_stride=convolution_stride
        )
        if node is not None:
            node.macCost += cost
            op_id = 0
            while True:
                if "{0}_{1}".format(name, op_id) in node.opMacCostsDict:
                    op_id += 1
                    continue
                break
            node.opMacCostsDict["{0}_{1}".format(name, op_id)] = cost
        x_hat = tf.nn.conv2d(x, kernel, strides, padding=padding)
        if bias is not None:
            x_hat = tf.nn.bias_add(x_hat, bias)
        return x_hat

    @staticmethod
    def fc_layer(x, W, b, node, name="fc_op"):
        assert len(x.get_shape().as_list()) == 2
        assert len(W.get_shape().as_list()) == 2
        x_hat = tf.matmul(x, W) + b
        num_of_input_channels = x.get_shape().as_list()[1]
        num_of_output_channels = W.get_shape().as_list()[1]
        cost = UtilityFuncs.calculate_mac_of_computation(num_of_input_channels=num_of_input_channels,
                                                         height_of_input_map=1,
                                                         width_of_input_map=1,
                                                         height_of_filter=1,
                                                         width_of_filter=1,
                                                         num_of_output_channels=num_of_output_channels,
                                                         convolution_stride=1,
                                                         type="fc")
        if node is not None:
            node.macCost += cost
            op_id = 0
            while True:
                if "{0}_{1}".format(name, op_id) in node.opMacCostsDict:
                    op_id += 1
                    continue
                break
            node.opMacCostsDict["{0}_{1}".format(name, op_id)] = cost
        return x_hat

    # OK for MultiGPU
    def build_tree(self):
        # Create itself
        curr_index = 0
        is_leaf = 0 == (self.depth - 1)
        root_node = Node(index=curr_index, depth=0, is_root=True, is_leaf=is_leaf)
        threshold_name = self.get_variable_name(name="threshold", node=root_node)
        root_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
        softmax_decay_name = self.get_variable_name(name="softmax_decay", node=root_node)
        root_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
        self.dagObject.add_node(node=root_node)
        self.nodes[curr_index] = root_node
        d = deque()
        d.append(root_node)
        # Create children if not leaf
        while len(d) > 0:
            # Dequeue
            curr_node = d.popleft()
            if not curr_node.isLeaf:
                for i in range(self.degreeList[curr_node.depth]):
                    new_depth = curr_node.depth + 1
                    is_leaf = new_depth == (self.depth - 1)
                    curr_index += 1
                    child_node = Node(index=curr_index, depth=new_depth, is_root=False, is_leaf=is_leaf)
                    if not child_node.isLeaf:
                        threshold_name = self.get_variable_name(name="threshold", node=child_node)
                        child_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
                        softmax_decay_name = self.get_variable_name(name="softmax_decay", node=child_node)
                        child_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
                    self.nodes[curr_index] = child_node
                    self.dagObject.add_edge(parent=curr_node, child=child_node)
                    d.append(child_node)
        nodes_per_level_dict = {}
        for node in self.nodes.values():
            if node.depth not in nodes_per_level_dict:
                nodes_per_level_dict[node.depth] = []
            nodes_per_level_dict[node.depth].append(node)
        for level in nodes_per_level_dict.keys():
            self.orderedNodesPerLevel[level] = sorted(nodes_per_level_dict[level], key=lambda n: n.index)

    @staticmethod
    def get_mock_tree(degree_list, network_name):
        tree = FastTreeNetwork(node_build_funcs=None, grad_func=None, hyperparameter_func=None,
                               residue_func=None, summary_func=None, degree_list=degree_list, dataset=None,
                               network_name=network_name)
        # Build the tree topologically and create the Tensorflow placeholders
        tree.build_tree()
        # Build symbolic networks
        tree.topologicalSortedNodes = tree.dagObject.get_topological_sort()
        tree.networkName = network_name
        return tree

    def prepare_evaluation_dictionary(self):
        # Prepare tensors to evaluate
        for node in self.topologicalSortedNodes:
            # if node.isLeaf:
            #     continue
            f_output = node.fOpsList[-1]
            self.evalDict[UtilityFuncs.get_variable_name(name="F", node=node)] = f_output
            # H
            if len(node.hOpsList) > 0:
                h_output = node.hOpsList[-1]
                self.evalDict[UtilityFuncs.get_variable_name(name="H", node=node)] = h_output
            # Activations
            for k, v in node.activationsDict.items():
                self.evalDict[UtilityFuncs.get_variable_name(name="activation_from_{0}".format(k), node=node)] = v
            # Decision masks
            for k, v in node.maskTensors.items():
                self.evalDict[UtilityFuncs.get_variable_name(name="{0}".format(v.name), node=node)] = v
            # Evaluation outputs
            for k, v in node.evalDict.items():
                self.evalDict[k] = v
            # Label outputs
            if node.labelTensor is not None:
                self.evalDict[UtilityFuncs.get_variable_name(name="label_tensor", node=node)] = node.labelTensor
                # Sample indices
                self.evalDict[UtilityFuncs.get_variable_name(name="indices_tensor", node=node)] = node.indicesTensor
            # One Hot Label outputs
            if node.oneHotLabelTensor is not None:
                self.evalDict[UtilityFuncs.get_variable_name(name="one_hot_label_tensor", node=node)] = \
                    node.oneHotLabelTensor
            if node.filteredMask is not None:
                self.evalDict[UtilityFuncs.get_variable_name(name="filteredMask", node=node)] = node.filteredMask
            # Batch Indices
            if node.batchIndicesTensor is not None:
                self.evalDict[UtilityFuncs.get_variable_name(name="batchIndicesTensor", node=node)] = \
                    node.batchIndicesTensor
            # # F
            # f_output = node.fOpsList[-1]
            # self.evalDict["Node{0}_F".format(node.index)] = f_output
            # # H
            # if len(node.hOpsList) > 0:
            #     h_output = node.hOpsList[-1]
            #     self.evalDict["Node{0}_H".format(node.index)] = h_output
            # # Activations
            # for k, v in node.activationsDict.items():
            #     self.evalDict["Node{0}_activation_from_{1}".format(node.index, k)] = v
            # # Decision masks
            # for k, v in node.maskTensors.items():
            #     self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
            # # Evaluation outputs
            # for k, v in node.evalDict.items():
            #     self.evalDict[k] = v
            # # Label outputs
            # if node.labelTensor is not None:
            #     self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
            #     # Sample indices
            #     self.evalDict["Node{0}_indices_tensor".format(node.index)] = node.indicesTensor
            # # One Hot Label outputs
            # if node.oneHotLabelTensor is not None:
            #     self.evalDict["Node{0}_one_hot_label_tensor".format(node.index)] = node.oneHotLabelTensor
            # if node.filteredMask is not None:
            #     self.evalDict["Node{0}_filteredMask".format(node.index)] = node.filteredMask
        self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}

    def build_optimizer(self):
        # Build optimizer
        self.globalCounter = tf.Variable(0, trainable=False)
        boundaries = [tpl[0] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule]
        values = [GlobalConstants.INITIAL_LR]
        values.extend([tpl[1] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule])
        self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
        # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
        with tf.control_dependencies(self.extra_update_ops):
            self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.finalLoss,
                                                                                         global_step=self.globalCounter)
            # self.optimizer = tf.train.AdamOptimizer().minimize(self.finalLoss, global_step=self.globalCounter)

    def build_network(self):
        # Build the tree topologically and create the Tensorflow placeholders
        self.build_tree()
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Disable some properties if we are using a baseline
        if self.isBaseline:
            GlobalConstants.USE_INFO_GAIN_DECISION = False
            GlobalConstants.USE_CONCAT_TRICK = False
            GlobalConstants.USE_PROBABILITY_THRESHOLD = False
        # Build all symbolic networks in each node
        for node in self.topologicalSortedNodes:
            print("Building Node {0}".format(node.index))
            self.nodeBuildFuncs[node.depth](network=self, node=node)
        # Build the residue loss
        # self.build_residue_loss()
        # Record all variables into the variable manager (For backwards compatibility)
        # self.variableManager.get_all_node_variables()
        self.dbName = DbLogger.log_db_path[DbLogger.log_db_path.rindex("/") + 1:]
        print(self.dbName)
        self.nodeCosts = {node.index: node.macCost for node in self.topologicalSortedNodes}
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        if not GlobalConstants.USE_MULTI_GPU:
            self.build_optimizer()
        self.prepare_evaluation_dictionary()

    def apply_decision(self, node, branching_feature):
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = tf.layers.batch_normalization(inputs=branching_feature,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain,
                                                                               tf.bool))
        ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
        node_degree = self.degreeList[node.depth]
        if GlobalConstants.USE_MULTI_GPU:
            # MultiGPU OK
            hyperplane_weights = UtilityFuncs.create_variable(
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
                shape=[ig_feature_size, node_degree],
                dtype=GlobalConstants.DATA_TYPE,
                initializer=tf.truncated_normal(
                    [ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                    dtype=GlobalConstants.DATA_TYPE))
            # MultiGPU OK
            hyperplane_biases = UtilityFuncs.create_variable(
                name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
                shape=[node_degree],
                dtype=GlobalConstants.DATA_TYPE,
                initializer=tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE))
        else:
            hyperplane_weights = tf.Variable(
                tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
            hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
        activations = FastTreeNetwork.fc_layer(x=branching_feature, W=hyperplane_weights, b=hyperplane_biases,
                                               node=node)
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / node.softmaxDecay
        p_n_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = node.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
        node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="branch_probs", node=node)] = p_n_given_x
        arg_max_indices = tf.argmax(p_n_given_x, axis=1)
        child_nodes = self.dagObject.children(node=node)
        child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
        for index in range(len(child_nodes)):
            child_node = child_nodes[index]
            child_index = child_node.index
            branch_prob = p_n_given_x[:, index]
            mask_with_threshold = tf.reshape(tf.greater_equal(x=branch_prob, y=node.probabilityThreshold,
                                                              name="Mask_with_threshold_{0}".format(child_index)), [-1])
            mask_without_threshold = tf.reshape(tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64),
                                                         name="Mask_without_threshold_{0}".format(child_index)), [-1])
            if GlobalConstants.USE_MULTI_GPU:
                with tf.device("/device:CPU:0"):
                    mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            else:
                mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            node.maskTensors[child_index] = mask_tensor
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors

    def apply_decision_with_unified_batch_norm(self, node, branching_feature):
        masked_branching_feature = tf.boolean_mask(branching_feature, node.filteredMask)
        if GlobalConstants.USE_MULTI_GPU:
            normed_x = CustomBatchNormAlgorithms.batch_norm_multi_gpu_v2(x=branching_feature,
                                                                         masked_x=masked_branching_feature,
                                                                         network=self, node=node,
                                                                         momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                                         is_training=self.isTrain)
            ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
            node_degree = self.degreeList[node.depth]
            # MultiGPU OK
            hyperplane_weights = UtilityFuncs.create_variable(
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
                shape=[ig_feature_size, node_degree],
                dtype=GlobalConstants.DATA_TYPE,
                initializer=tf.truncated_normal(
                    [ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                    dtype=GlobalConstants.DATA_TYPE))
            # MultiGPU OK
            hyperplane_biases = UtilityFuncs.create_variable(
                name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
                shape=[node_degree],
                dtype=GlobalConstants.DATA_TYPE,
                initializer=tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE))
        else:
            normed_x = CustomBatchNormAlgorithms.masked_batch_norm(x=branching_feature,
                                                                   masked_x=masked_branching_feature,
                                                                   network=self, node=node,
                                                                   momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                                   iteration=self.iterationHolder,
                                                                   is_training_phase=self.isTrain)
            ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
            node_degree = self.degreeList[node.depth]
            hyperplane_weights = tf.Variable(
                tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
            hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
        activations = FastTreeNetwork.fc_layer(x=normed_x, W=hyperplane_weights, b=hyperplane_biases,
                                               node=node)
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / node.softmaxDecay
        p_n_given_x = tf.nn.softmax(decayed_activation)
        p_n_given_x_masked = tf.boolean_mask(p_n_given_x, node.filteredMask)
        p_c_given_x = node.oneHotLabelTensor
        p_c_given_x_masked = tf.boolean_mask(p_c_given_x, node.filteredMask)
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x_masked, p_c_given_x_2d=p_c_given_x_masked,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
        node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="branch_probs", node=node)] = p_n_given_x
        node.evalDict[self.get_variable_name(name="branch_probs_masked", node=node)] = p_n_given_x_masked
        node.evalDict[self.get_variable_name(name="p(c|x)", node=node)] = p_c_given_x
        node.evalDict[self.get_variable_name(name="p(c|x)_masked", node=node)] = p_c_given_x_masked
        arg_max_indices = tf.argmax(p_n_given_x, axis=1)
        child_nodes = self.dagObject.children(node=node)
        child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
        for index in range(len(child_nodes)):
            child_node = child_nodes[index]
            child_index = child_node.index
            branch_prob = p_n_given_x[:, index]
            mask_with_threshold = tf.reshape(tf.greater_equal(x=branch_prob, y=node.probabilityThreshold,
                                                              name="Mask_with_threshold_{0}".format(child_index)), [-1])
            mask_without_threshold = tf.reshape(tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64),
                                                         name="Mask_without_threshold_{0}".format(child_index)), [-1])
            mask_without_threshold = tf.logical_and(mask_without_threshold, node.filteredMask)
            if GlobalConstants.USE_MULTI_GPU:
                with tf.device("/device:CPU:0"):
                    mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            else:
                mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            node.maskTensors[child_index] = mask_tensor
            node.masksWithoutThreshold[child_index] = mask_without_threshold
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors
            node.evalDict[self.get_variable_name(name="masksWithoutThreshold", node=node)] = node.masksWithoutThreshold

    # MultiGPU OK
    def mask_input_nodes(self, node):
        if node.isRoot:
            node.batchIndicesTensor = tf.range(0, self.batchSizeTf, 1)
            node.labelTensor = self.labelTensor
            node.indicesTensor = self.indicesTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            # node.filteredMask = tf.constant(value=True, dtype=tf.bool, shape=(GlobalConstants.BATCH_SIZE, ))
            node.filteredMask = self.filteredMask
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            return None, None
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            mask_tensor = parent_node.maskTensors[node.index]
            if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                mask_without_threshold = parent_node.masksWithoutThreshold[node.index]
            if GlobalConstants.USE_MULTI_GPU:
                with tf.device("/device:CPU:0"):
                    mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
                                           tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            else:
                mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
                                       tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            if GlobalConstants.USE_MULTI_GPU:
                with tf.device("/device:CPU:0"):
                    node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            else:
                node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # Mask all inputs: F channel, H  channel, activations from ancestors, labels
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            if GlobalConstants.USE_MULTI_GPU:
                with tf.device("/device:CPU:0"):
                    node.batchIndicesTensor = tf.boolean_mask(parent_node.batchIndicesTensor, mask_tensor)
                    node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
                    node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
                    node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
                    if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                        node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            else:
                node.batchIndicesTensor = tf.boolean_mask(parent_node.batchIndicesTensor, mask_tensor)
                node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
                node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
                node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
                if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                    node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            if GlobalConstants.USE_SCALED_GRADIENTS:
                parent_child_count = len(self.dagObject.children(node=parent_node))
                scale = 1.0 / parent_child_count
                parent_F = scale * parent_F + (1 - scale) * tf.stop_gradient(parent_F)
                parent_H = scale * parent_H + (1 - scale) * tf.stop_gradient(parent_H)
            return parent_F, parent_H

    def make_loss(self, node, logits):
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                                   logits=logits)
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
        node.lossList.append(loss)
        return loss

    # MultiGPU OK
    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=softmax_weights, b=softmax_biases, node=node)
        node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
        _ = self.make_loss(node=node, logits=logits)
        return final_feature, logits

    def get_run_ops(self):
        run_ops = [self.optimizer, self.learningRate, self.sampleCountTensors, self.isOpenTensors,
                   self.infoGainDicts]
        # run_ops = [self.learningRate, self.sampleCountTensors, self.isOpenTensors,
        #            self.infoGainDicts]
        return run_ops

    def verbose_update(self, eval_dict):
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            for level in range(self.depth):
                if level == 0:
                    continue
                level_nodes = [node for node in self.topologicalSortedNodes if node.depth == level]
                sum_of_samples = 0
                for node in level_nodes:
                    filtered_mask = eval_dict["Node{0}_filteredMask".format(node.index)]
                    sum_of_samples += np.sum(filtered_mask)
                if sum_of_samples != GlobalConstants.BATCH_SIZE:
                    print("ERR")
                assert sum_of_samples == GlobalConstants.BATCH_SIZE

    def update_params(self, sess, dataset, epoch, iteration):
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.BATCH_SIZE
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.CURR_BATCH_SIZE)
        if minibatch is None:
            return TrainingUpdateResult(lr=None, sample_counts=None, is_open_indicators=None)
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                           is_train=True, use_masking=True)
        # Prepare result tensors to collect
        run_ops = self.get_run_ops()
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        print("Before Update Iteration:{0}".format(iteration))
        results = sess.run(run_ops, feed_dict=feed_dict)
        print("After Update Iteration:{0}".format(iteration))
        lr = results[1]
        sample_counts = results[2]
        is_open_indicators = results[3]
        # Unit Test for Unified Batch Normalization
        if GlobalConstants.USE_VERBOSE:
            self.verbose_update(eval_dict=results[-1])
            update_results = TrainingUpdateResult(lr=lr, sample_counts=sample_counts,
                                                  is_open_indicators=is_open_indicators, eval_dict=results[-1])
        else:
            # Unit Test for Unified Batch Normalization
            update_results = TrainingUpdateResult(lr=lr, sample_counts=sample_counts,
                                                  is_open_indicators=is_open_indicators)
        return update_results

    def eval_network(self, sess, dataset, use_masking):
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.EVAL_BATCH_SIZE
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.CURR_BATCH_SIZE)
        if minibatch is None:
            return None, None
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=1000000, use_threshold=False,
                                           is_train=False, use_masking=use_masking)
        eval_filtered = {k: v for k, v in self.evalDict.items() if v is not None}
        results = sess.run(eval_filtered, feed_dict)
        # path_dict = {k: v for k, v in results.items() if "all_paths_used" in k}
        # if not all(list(path_dict.values())):
        #     print("X")
        # for k, v in results.items():
        #     if "final_feature_mag" in k:
        #         print("{0}={1}".format(k, v))
        return results, minibatch

    def eval_minibatch(self, sess, minibatch, use_masking):
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=1000000, use_threshold=False,
                                           is_train=False, use_masking=use_masking)
        eval_filtered = {k: v for k, v in self.evalDict.items() if v is not None}
        results = sess.run(eval_filtered, feed_dict)
        # for k, v in results.items():
        #     if "final_feature_mag" in k:
        #         print("{0}={1}".format(k, v))
        return results, minibatch

    def collect_outputs_into_collection(self, collection, output_names, node, results):
        for output_name in output_names:
            if "original_samples" == output_name:
                continue
            key_name = self.get_variable_name(name=output_name, node=node)
            if key_name not in results:
                continue
            output_arr = results[self.get_variable_name(name=output_name, node=node)]
            UtilityFuncs.concat_to_np_array_dict_v2(dct=collection[output_name], key=node.index, array=output_arr)

    def collect_eval_results_from_network(self,
                                          sess,
                                          dataset,
                                          dataset_type,
                                          use_masking,
                                          leaf_node_collection_names,
                                          inner_node_collections_names):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_node_collections = {}
        inner_node_collections = {}
        for output_name in leaf_node_collection_names:
            leaf_node_collections[output_name] = {}
        for output_name in inner_node_collections_names:
            inner_node_collections[output_name] = {}
        while True:
            results, minibatch = self.eval_network(sess=sess, dataset=dataset, use_masking=use_masking)
            if results is not None:
                for node in self.topologicalSortedNodes:
                    if not node.isLeaf:
                        self.collect_outputs_into_collection(collection=inner_node_collections,
                                                             output_names=inner_node_collections_names,
                                                             node=node,
                                                             results=results)
                    else:
                        self.collect_outputs_into_collection(collection=leaf_node_collections,
                                                             output_names=leaf_node_collection_names,
                                                             node=node,
                                                             results=results)
                # Add the input data as well.
                if "original_samples" in set(inner_node_collections_names):
                    UtilityFuncs.concat_to_np_array_dict_v2(dct=inner_node_collections["original_samples"], key=0,
                                                            array=minibatch.samples)
            if dataset.isNewEpoch:
                break
        for collections in [leaf_node_collections, inner_node_collections]:
            for output_name, nodes_arr_dict in collections.items():
                for node_idx in nodes_arr_dict.keys():
                    if np.isscalar(nodes_arr_dict[node_idx][0]):
                        continue
                    nodes_arr_dict[node_idx] = np.concatenate(nodes_arr_dict[node_idx], axis=0)
        return leaf_node_collections, inner_node_collections

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = {self.dataTensor: minibatch.samples,
                     self.labelTensor: minibatch.labels,
                     self.indicesTensor: minibatch.indices,
                     self.oneHotLabelTensor: minibatch.one_hot_labels,
                     # self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
                     self.useThresholding: int(use_threshold),
                     # self.isDecisionPhase: int(is_decision_phase),
                     self.isTrain: int(is_train),
                     self.useMasking: int(use_masking),
                     self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
                     self.iterationHolder: iteration,
                     self.filteredMask: np.ones((GlobalConstants.CURR_BATCH_SIZE,), dtype=bool),
                     self.batchSizeTf: GlobalConstants.CURR_BATCH_SIZE}
        if is_train:
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB
            if not self.isBaseline:
                self.get_probability_thresholds(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=True)
            if self.modeTracker.isCompressed:
                self.get_label_mappings(feed_dict=feed_dict)
        else:
            feed_dict[self.classificationDropoutKeepProb] = 1.0
            if not self.isBaseline:
                self.get_probability_thresholds(feed_dict=feed_dict, iteration=1000000, update=False)
                self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=False)
                feed_dict[self.decisionDropoutKeepProb] = 1.0
        return feed_dict

    def set_training_parameters(self):
        pass

    def set_hyperparameters(self, **kwargs):
        pass

    def get_explanation_string(self):
        explanation = ""
        explanation += "TOTAL_EPOCH_COUNT:{0}\n".format(GlobalConstants.TOTAL_EPOCH_COUNT)
        explanation += "EPOCH_COUNT:{0}\n".format(GlobalConstants.EPOCH_COUNT)
        explanation += "EPOCH_REPORT_PERIOD:{0}\n".format(GlobalConstants.EPOCH_REPORT_PERIOD)
        explanation += "BATCH_SIZE:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "EVAL_BATCH_SIZE:{0}\n".format(GlobalConstants.EVAL_BATCH_SIZE)
        explanation += "USE_MULTI_GPU:{0}\n".format(GlobalConstants.USE_MULTI_GPU)
        explanation += "USE_SAMPLING_CIGN:{0}\n".format(GlobalConstants.USE_SAMPLING_CIGN)
        explanation += "USE_RANDOM_SAMPLING:{0}\n".format(GlobalConstants.USE_RANDOM_SAMPLING)
        explanation += "USE_SCALED_GRADIENTS:{0}\n".format(GlobalConstants.USE_SCALED_GRADIENTS)
        explanation += "LR SCHEDULE:{0}\n".format(GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation())
        leaf_node = [node for node in self.topologicalSortedNodes if node.isLeaf][0]
        root_to_leaf_path = self.dagObject.ancestors(node=leaf_node)
        root_to_leaf_path.append(leaf_node)
        path_mac_cost = sum([node.macCost for node in root_to_leaf_path])
        explanation += "Mac Cost:{0}\n".format(path_mac_cost)
        explanation += "Mac Cost per Nodes:{0}\n".format(self.nodeCosts)
        explanation += "Optimizer:{0}".format(self.optimizer)
        return explanation

    def print_iteration_info(self, iteration_counter, update_results):
        lr = update_results.lr
        sample_counts = update_results.sampleCounts
        is_open_indicators = update_results.isOpenIndicators
        print("Iteration:{0}".format(iteration_counter))
        print("Lr:{0}".format(lr))
        # Print sample counts (classification)
        sample_count_str = "Classification:   "
        for k, v in sample_counts.items():
            sample_count_str += "[{0}={1}]".format(k, v)
        print(sample_count_str)
        # Print node open indicators
        indicator_str = ""
        for k, v in is_open_indicators.items():
            indicator_str += "[{0}={1}]".format(k, v)
        print(indicator_str)

    # Sample from categorical distribution using Gumbel-Max trick
    @staticmethod
    def sample_from_categorical(probs, batch_size, category_count):
        uniform = tf.distributions.Uniform(low=0.0, high=1.0)
        uniform_sample = uniform.sample(sample_shape=(tf.cast(batch_size, tf.int32), category_count))
        gumbel_sample = -1.0 * tf.log(-1.0 * tf.log(uniform_sample))
        log_probs = tf.log(probs)
        gumbel_max = gumbel_sample + log_probs
        selected_indices = tf.cast(tf.argmax(gumbel_max, axis=1), tf.int32)
        return selected_indices

    @staticmethod
    def sample_from_categorical_v2(probs):
        uniform = tf.distributions.Uniform(low=0.0, high=1.0)
        prob_shape = tf.shape(probs)
        batch_size = tf.gather_nd(prob_shape, [0])
        category_count = tf.gather_nd(prob_shape, [1])
        uniform_sample = uniform.sample(sample_shape=(tf.cast(batch_size, tf.int32),
                                                      tf.cast(category_count, tf.int32)))
        gumbel_sample = -1.0 * tf.log(-1.0 * tf.log(uniform_sample))
        log_probs = tf.log(probs)
        gumbel_max = gumbel_sample + log_probs
        selected_indices = tf.cast(tf.argmax(gumbel_max, axis=1), tf.int32)
        return selected_indices

    def get_checkpoint_path(self, run_id, iteration):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.join(curr_path, ".."), ".."),
                                                                   "saved_training_data"),
                                                      "checkpoint_{0}_run_{1}_iteration_{2}".format(self.networkName,
                                                                                                    run_id, iteration)))
        checkpoint_path = os.path.abspath(os.path.join(directory_path, "model.ckpt"))
        return directory_path, checkpoint_path

    @staticmethod
    def get_routing_info_path(network_name, run_id, iteration, data_type):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        if data_type != "":
            directory_path = os.path.abspath(
                os.path.join(os.path.join(os.path.join(os.path.join(curr_path, ".."), ".."),
                                          "saved_training_data"),
                             "{0}_run_{1}_iteration_{2}_{3}".format(network_name, run_id,
                                                                    iteration, data_type)))
        else:
            directory_path = os.path.abspath(
                os.path.join(os.path.join(os.path.join(os.path.join(curr_path, ".."), ".."),
                                          "saved_training_data"),
                             "{0}_run_{1}_iteration_{2}".format(network_name,
                                                                run_id, iteration)))
        return directory_path

    def save_model(self, sess, run_id, iteration):
        directory_path, checkpoint_path = self.get_checkpoint_path(run_id=run_id, iteration=iteration)
        os.mkdir(directory_path)
        self.saver.save(sess, checkpoint_path)

    def load_model(self, sess, run_id, iteration):
        _, checkpoint_path = self.get_checkpoint_path(run_id=run_id, iteration=iteration)
        saved_vars = checkpoint_utils.list_variables(checkpoint_dir=checkpoint_path)
        all_vars = tf.global_variables()
        for var in all_vars:
            if "Adam" in var.name:
                continue
            source_array = checkpoint_utils.load_variable(checkpoint_dir=checkpoint_path, name=var.name)
            tf.assign(var, source_array).eval(session=sess)
        print("X")

    def save_routing_info(self, sess, run_id, iteration, dataset, dataset_type):
        dict_of_data_dicts = {}
        data_type = "test" if dataset_type == DatasetTypes.test else "training"
        directory_path = FastTreeNetwork.get_routing_info_path(network_name=self.networkName,
                                                               run_id=run_id, iteration=iteration,
                                                               data_type=data_type)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        inner_node_outputs = set(GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT)
        inner_node_outputs.add("original_samples")
        leaf_node_outputs = set(GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT)
        leaf_node_collections, inner_node_collections = \
            self.collect_eval_results_from_network(sess=sess, dataset=dataset, dataset_type=dataset_type,
                                                   use_masking=False,
                                                   leaf_node_collection_names=leaf_node_outputs,
                                                   inner_node_collections_names=inner_node_outputs)
        npz_file_name = os.path.abspath(os.path.join(directory_path, "tree_type"))
        UtilityFuncs.save_npz(file_name=npz_file_name, arr_dict={"tree_type": np.array(self.degreeList)})
        for output_names, collection in zip([inner_node_outputs, leaf_node_outputs],
                                            [inner_node_collections, leaf_node_collections]):
            for output_name in output_names:
                arr_dict = collection[output_name]
                dict_of_data_dicts[output_name] = arr_dict
                string_arr_dict = {"{0}".format(k): v for k, v in arr_dict.items()}
                npz_file_name = os.path.abspath(os.path.join(directory_path, output_name))
                UtilityFuncs.save_npz(file_name=npz_file_name, arr_dict=string_arr_dict)
        label_data = dict_of_data_dicts["label_tensor"]
        label_list = list(label_data.values())[0]
        assert all([np.array_equal(label_list, arr) for idx, arr in label_data.items()])
        dict_of_data_dicts["nodeCosts"] = self.nodeCosts
        pickle.dump(self.nodeCosts, open(os.path.abspath(os.path.join(directory_path, "nodeCosts.sav")), "wb"))
        for node in self.topologicalSortedNodes:
            pickle.dump(node.opMacCostsDict,
                        open(
                            os.path.abspath(
                                os.path.join(directory_path, "node_{0}_opMacCosts.sav".format(node.index))), "wb"))
            dict_of_data_dicts["node_{0}_opMacCosts".format(node.index)] = node.opMacCostsDict
        routing_data = RoutingDataset(label_list=label_list, dict_of_data_dicts=dict_of_data_dicts)
        return routing_data

    def load_routing_info(self,
                          run_id,
                          iteration,
                          data_type,
                          output_names=None):
        directory_path = FastTreeNetwork.get_routing_info_path(run_id=run_id, iteration=iteration,
                                                               network_name=self.networkName,
                                                               data_type=data_type)
        # Assert that the tree architecture is compatible with the loaded info
        npz_file_name = os.path.abspath(os.path.join(directory_path, "tree_type"))
        degree_list = UtilityFuncs.load_npz(file_name=npz_file_name)
        assert np.array_equal(np.array(self.degreeList), degree_list["tree_type"])
        dict_of_data_dicts = {}
        if output_names is None:
            all_output_names = set(GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT)
            all_output_names = all_output_names.union(set(GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT))
            all_output_names.add("original_samples")
        else:
            all_output_names = set(output_names)
            all_output_names.add("label_tensor")
        for output_name in all_output_names:
            try:
                npz_file_name = os.path.abspath(os.path.join(directory_path, output_name))
                dict_read = UtilityFuncs.load_npz(file_name=npz_file_name)
                data_dict = {int(k): v for k, v in dict_read.items()}
                dict_of_data_dicts[output_name] = data_dict
            except:
                print("Data for {0} does not exist.".format(output_name))
        assert "label_tensor" in dict_of_data_dicts
        label_data = dict_of_data_dicts["label_tensor"]
        label_list = list(label_data.values())[0]
        # Check the integrity of all the loaded data.
        assert all([np.array_equal(label_list, arr) for idx, arr in label_data.items()])
        for k, data_dict in dict_of_data_dicts.values():
            for arr in data_dict.values():
                assert arr.shape[0] == label_list.shape[0]
        # Load node cost information.
        try:
            self.nodeCosts = pickle.load(open(os.path.abspath(os.path.join(directory_path, "nodeCosts.sav")), 'rb'))
            dict_of_data_dicts["nodeCosts"] = self.nodeCosts
        except:
            print("Node Cost data does not exist.")
        try:
            for node in self.topologicalSortedNodes:
                node.opMacCostsDict = pickle.load(open(os.path.abspath(os.path.join(directory_path,
                                                                                    "node_{0}_opMacCosts.sav"
                                                                                    .format(node.index))), 'rb'))
                dict_of_data_dicts["node_{0}_opMacCosts".format(node.index)] = node.opMacCostsDict
        except:
            print("Op Mac Costs data does not exist.")
        routing_data = RoutingDataset(label_list=label_list, dict_of_data_dicts=dict_of_data_dicts)
        return routing_data

    # Unit Test
    def test_save_load(self, sess, run_id, iteration, dataset, dataset_type):
        data_type = "test" if dataset_type == DatasetTypes.test else "training"
        routing_data_save = self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                                   dataset=dataset, dataset_type=dataset_type)
        routing_data_load = self.load_routing_info(run_id=run_id, iteration=iteration, data_type=data_type)
        assert routing_data_save == routing_data_load

    @staticmethod
    def calculate_information_gain(info_gain_dict, kv_rows, dataset_type, run_id, iteration):
        # Measure Information Gain
        total_info_gain = 0.0
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
            total_info_gain -= avg_info_gain
            kv_rows.append((run_id, iteration, "Dataset:{0} IG:{1}".format(dataset_type, k), avg_info_gain))
        kv_rows.append((run_id, iteration, "Dataset:{0} Total IG".format(dataset_type), total_info_gain))

    @staticmethod
    def measure_branch_probs(branch_probs, run_id, iteration, dataset_type, kv_rows):
        # Measure Branching Probabilities
        for k, v in branch_probs.items():
            p_n = np.mean(v, axis=0)
            arg_max_arr = np.argmax(v, axis=1)
            max_counts = {i: np.sum(arg_max_arr == i) for i in range(p_n.shape[0])}
            print("Argmax counts:{0}".format(max_counts))
            for branch in range(p_n.shape[0]):
                print("{0} p_{1}({2})={3}".format(dataset_type, k, branch, p_n[branch]))
                kv_rows.append((run_id, iteration, "{0} p_{1}({2})".format(dataset_type, k, branch),
                                np.asscalar(p_n[branch])))

    @staticmethod
    def calculate_branch_probability_histograms(branch_probs):
        for k, v in branch_probs.items():
            # Interval analysis
            print("Node:{0}".format(k))
            bin_size = 0.1
            for j in range(v.shape[1]):
                histogram = {}
                for i in range(v.shape[0]):
                    prob = v[i, j]
                    bin_id = int(prob / bin_size)
                    if bin_id not in histogram:
                        histogram[bin_id] = 0
                    histogram[bin_id] += 1
                sorted_histogram = sorted(list(histogram.items()), key=lambda e: e[0], reverse=False)
                print(histogram)

    def label_distribution_analysis(self,
                                    run_id,
                                    iteration,
                                    kv_rows,
                                    leaf_true_labels_dict,
                                    dataset,
                                    dataset_type):
        label_count = dataset.get_label_count()
        label_distribution = np.zeros(shape=(label_count,))
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if node.index not in leaf_true_labels_dict:
                continue
            true_labels = leaf_true_labels_dict[node.index]
            for l in range(label_count):
                label_distribution[l] = np.sum(true_labels == l)
                kv_rows.append((run_id, iteration, "{0} Leaf:{1} True Label:{2}".
                                format(dataset_type, node.index, l), np.asscalar(label_distribution[l])))

    def calculate_accuracy(self, sess, dataset, dataset_type, run_id, iteration,
                           posterior_entry_name="posterior_probs"):
        kv_rows = []
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        inner_node_outputs = ["info_gain", "branch_probs", "activations"]
        leaf_node_outputs = [posterior_entry_name, "label_tensor"]
        t0 = time.time()
        leaf_node_collections, inner_node_collections = \
            self.collect_eval_results_from_network(sess=sess, dataset=dataset, dataset_type=dataset_type,
                                                   use_masking=True,
                                                   leaf_node_collection_names=leaf_node_outputs,
                                                   inner_node_collections_names=inner_node_outputs)
        t1 = time.time()
        print(t1 - t0)
        info_gain_dict = inner_node_collections["info_gain"]
        branch_probs_dict = inner_node_collections["branch_probs"]
        leaf_true_labels_dict = leaf_node_collections["label_tensor"]
        posteriors_dict = leaf_node_collections[posterior_entry_name]
        # Measure Information Gain
        FastTreeNetwork.calculate_information_gain(info_gain_dict=info_gain_dict, kv_rows=kv_rows,
                                                   dataset_type=dataset_type, run_id=run_id, iteration=iteration)
        # Measure Branching Probabilities
        FastTreeNetwork.measure_branch_probs(run_id=run_id, iteration=iteration, dataset_type=dataset_type,
                                             branch_probs=branch_probs_dict, kv_rows=kv_rows)
        # Measure The Histogram of Branching Probabilities
        FastTreeNetwork.calculate_branch_probability_histograms(branch_probs=branch_probs_dict)
        # Measure Label Distribution
        self.label_distribution_analysis(run_id=run_id, iteration=iteration, kv_rows=kv_rows,
                                         leaf_true_labels_dict=leaf_true_labels_dict,
                                         dataset=dataset, dataset_type=dataset_type)
        # Measure accuracy
        total_correct_count = 0
        total_sample_count = 0
        all_predicted = None
        all_truths = None
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            leaf_posteriors = posteriors_dict[node.index]
            true_labels = leaf_true_labels_dict[node.index]
            assert leaf_posteriors.shape[0] == true_labels.shape[0]
            predicted_labels = np.argmax(leaf_posteriors, axis=1)
            correct_count = np.sum(predicted_labels == true_labels)
            total_correct_count += correct_count
            total_sample_count += true_labels.shape[0]
            if all_predicted is None:
                all_predicted = predicted_labels
            else:
                all_predicted = np.concatenate([all_predicted, predicted_labels], axis=0)
            if all_truths is None:
                all_truths = true_labels
            else:
                all_truths = np.concatenate([all_truths, true_labels], axis=0)
            if true_labels.shape[0] > 0:
                print("Leaf {0}: Sample Count:{1} Accuracy:{2}".format(node.index, correct_count,
                                                                       float(correct_count) / float(
                                                                           true_labels.shape[0])))
            else:
                print("Leaf {0} is empty.".format(node.index))
        total_accuracy = float(total_correct_count) / float(total_sample_count)
        # Prepare the confusion matrix
        cm = confusion_matrix(y_true=all_truths, y_pred=all_predicted)
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(total_sample_count, total_accuracy))
        # Calculate modes
        self.modeTracker.calculate_modes(leaf_true_labels_dict=leaf_true_labels_dict,
                                         dataset=dataset, dataset_type=dataset_type, kv_rows=kv_rows,
                                         run_id=run_id, iteration=iteration)
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        return total_accuracy, cm

    def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
        # moving_results_1 = sess.run(moving_stat_vars)
        is_evaluation_epoch_at_report_period = \
            epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
            and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
        is_evaluation_epoch_before_ending = \
            epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
        if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
            training_accuracy, training_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy, validation_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy_corrected = 0.0
            if not self.isBaseline:
                validation_accuracy_corrected, validation_marginal_corrected = \
                    self.accuracyCalculator.calculate_accuracy_with_route_correction(
                        sess=sess, dataset=dataset,
                        dataset_type=DatasetTypes.test)
                if is_evaluation_epoch_before_ending:
                    # self.save_model(sess=sess, run_id=run_id, iteration=iteration)
                    t0 = time.time()
                    self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                           dataset=dataset, dataset_type=DatasetTypes.training)
                    t1 = time.time()
                    self.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
                                           dataset=dataset, dataset_type=DatasetTypes.test)
                    t2 = time.time()
                    print("t1-t0={0}".format(t1 - t0))
                    print("t2-t1={0}".format(t2 - t1))
                    # self.test_save_load(sess=sess, run_id=run_id, iteration=iteration,
                    #                     dataset=dataset, dataset_type=DatasetTypes.test)
            DbLogger.write_into_table(
                rows=[(run_id, iteration, epoch_id, training_accuracy,
                       validation_accuracy, validation_accuracy_corrected,
                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)

    def train(self, sess, dataset, run_id):
        iteration_counter = 0
        self.saver = tf.train.Saver()
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            while True:
                print("Iteration:{0}".format(iteration_counter))
                start_time = time.time()
                update_results = self.update_params(sess=sess,
                                                    dataset=dataset,
                                                    epoch=epoch_id,
                                                    iteration=iteration_counter)
                print("Update_results type new:{0}".format(update_results.__class__))
                if all([update_results.lr, update_results.sampleCounts, update_results.isOpenIndicators]):
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    self.print_iteration_info(iteration_counter=iteration_counter, update_results=update_results)
                    iteration_counter += 1
                if dataset.isNewEpoch:
                    print("Epoch Time={0}".format(total_time))
                    self.calculate_model_performance(sess=sess, dataset=dataset, run_id=run_id, epoch_id=epoch_id,
                                                     iteration=iteration_counter)
                    break
