import tensorflow as tf
import numpy as np

from collections import deque

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node


class FastTreeMultiGpu(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list, dataset,
                 container_network, tower_id, tower_batch_size):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list, dataset)

    def build_network(self):
        # Build the tree topologically and create the Tensorflow placeholders
        # MultiGPU OK
        self.build_tree()
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Disable some properties if we are using a baseline
        if self.isBaseline:
            GlobalConstants.USE_INFO_GAIN_DECISION = False
            GlobalConstants.USE_CONCAT_TRICK = False
            GlobalConstants.USE_PROBABILITY_THRESHOLD = False
        # Build the symbolic network using the given variable scope and the provided device
        # MultiGPU OK
        # with tf.device(self.deviceStr):
        # Build all symbolic networks in each node
        for node in self.topologicalSortedNodes:
            print("Building Node {0}".format(node.index))
            self.nodeBuildFuncs[node.depth](node=node, network=self)
        # Build main classification loss
        # MultiGPU OK
        self.build_main_loss()
        # Build information gain loss
        # MultiGPU OK
        self.build_decision_loss()
        # Build regularization loss
        # MultiGPU OK
        self.build_regularization_loss()
        # Final Loss
        # MultiGPU OK
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        # MultiGPU OK
        self.prepare_evaluation_dictionary()

    # MultiGPU OK
    def apply_decision_with_unified_batch_norm(self, node, branching_feature):
        masked_branching_feature = tf.boolean_mask(branching_feature, node.filteredMask)
        # MultiGPU OK
        normed_x = CustomBatchNormAlgorithms.masked_batch_norm_multi_gpu(
            input_tensor=branching_feature,
            masked_input_tensor=masked_branching_feature,
            is_training=self.isTrain,
            momentum=GlobalConstants.BATCH_NORM_DECAY,
            network=self, node=node
        )
        # normed_x = CustomBatchNormAlgorithms.masked_batch_norm(x=branching_feature, masked_x=masked_branching_feature,
        #                                                        network=self, node=node,
        #                                                        momentum=GlobalConstants.BATCH_NORM_DECAY,
        #                                                        iteration=self.iterationHolder,
        #                                                        is_training_phase=self.isTrain)
        ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
        node_degree = self.degreeList[node.depth]
        # MultiGPU OK
        hyperplane_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
            shape=[ig_feature_size, node_degree],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
                [ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        # MultiGPU OK
        hyperplane_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
            shape=[node_degree],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE))
        activations = tf.matmul(normed_x, hyperplane_weights) + hyperplane_biases
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
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        node.evalDict[self.get_variable_name(name="p(n|x)_masked", node=node)] = p_n_given_x_masked
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
            with tf.device("/device:CPU:0"):
                mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            node.maskTensors[child_index] = mask_tensor
            node.masksWithoutThreshold[child_index] = mask_without_threshold
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors
            node.evalDict[self.get_variable_name(name="masksWithoutThreshold", node=node)] = node.masksWithoutThreshold

    # MultiGPU OK
    def apply_decision(self, node, branching_feature):
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = CustomBatchNormAlgorithms.batch_norm_multi_gpu_v2(
                input_tensor=branching_feature,
                is_training=self.isTrain,
                momentum=GlobalConstants.BATCH_NORM_DECAY,
                network=self, node=node
            )
        ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
        node_degree = self.degreeList[node.depth]
        # MultiGPU OK
        hyperplane_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
            shape=[ig_feature_size, node_degree],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
                [ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        # MultiGPU OK
        hyperplane_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
            shape=[node_degree],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE))
        activations = tf.matmul(branching_feature, hyperplane_weights) + hyperplane_biases
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
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
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
            with tf.device("/device:CPU:0"):
                mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            node.maskTensors[child_index] = mask_tensor
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors

    def mask_input_nodes(self, node):
        if node.isRoot:
            node.labelTensor = self.labelTensor
            node.indicesTensor = self.indicesTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            # node.filteredMask = tf.constant(value=True, dtype=tf.bool, shape=(GlobalConstants.BATCH_SIZE, ))
            node.filteredMask = self.filteredMask
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            mask_tensor = parent_node.maskTensors[node.index]
            if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                mask_without_threshold = parent_node.masksWithoutThreshold[node.index]
            with tf.device("/device:CPU:0"):
                mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
                                       tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            with tf.device("/device:CPU:0"):
                node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # Mask all inputs: F channel, H  channel, activations from ancestors, labels
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            with tf.device("/device:CPU:0"):
                if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                    node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            return parent_F, parent_H