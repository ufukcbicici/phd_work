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
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset,
                 container_network, tower_id, tower_batch_size, device_str):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        self.towerId = tower_id
        self.deviceStr = device_str
        self.towerBatchSize = tower_batch_size
        lower_bound = int(self.towerId * self.towerBatchSize)
        upper_bound = int((self.towerId + 1) * self.towerBatchSize)
        self.dataTensor = container_network.dataTensor[lower_bound:upper_bound]
        self.labelTensor = container_network.labelTensor[lower_bound:upper_bound]
        self.indicesTensor = container_network.indicesTensor[lower_bound:upper_bound]
        self.oneHotLabelTensor = container_network.oneHotLabelTensor[lower_bound:upper_bound]
        self.filteredMask = container_network.filteredMask[lower_bound:upper_bound]
        self.isTrain = container_network.isTrain

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
        # Build the symbolic network using the given variable scope and the provided device
        with tf.device(self.deviceStr):
            with tf.name_scope("tower_{0}".format(self.towerId)):
                # Build all symbolic networks in each node
                for node in self.topologicalSortedNodes:
                    self.nodeBuildFuncs[node.depth](node=node, network=self)

        # gpu_names = UtilityFuncs.get_available_devices()
        # self.towerCount = len(gpu_names)
        # assert GlobalConstants.BATCH_SIZE % self.towerCount == 0
        # self.towerBatchSize = GlobalConstants.BATCH_SIZE / self.towerCount
        # for tower_id in range(self.towerCount):
        #     self.currentTowerId = tower_id
        #     with tf.device('/gpu:%d' % tower_id):
        #         with tf.name_scope("tower_{0}".format(tower_id)):
        #             # Build all symbolic networks in each node
        #             for node in self.topologicalSortedNodes:
        #                 self.nodeBuildFuncs[node.depth](node=node, network=self)
        # # Build main classification loss
        # self.build_main_loss()

    # # Build information gain loss
    # self.build_decision_loss()
    # # Build regularization loss
    # self.build_regularization_loss()
    # # Final Loss
    # self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss + self.residueLoss
    # # Build optimizer
    # self.globalCounter = tf.Variable(0, trainable=False)
    # boundaries = [tpl[0] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule]
    # values = [GlobalConstants.INITIAL_LR]
    # values.extend([tpl[1] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule])
    # self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
    # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
    # # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
    # with tf.control_dependencies(self.extra_update_ops):
    #     self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.finalLoss,
    #                                                                                  global_step=self.globalCounter)
    # # Prepare tensors to evaluate
    # for node in self.topologicalSortedNodes:
    #     # if node.isLeaf:
    #     #     continue
    #     # F
    #     f_output = node.fOpsList[-1]
    #     self.evalDict["Node{0}_F".format(node.index)] = f_output
    #     # H
    #     if len(node.hOpsList) > 0:
    #         h_output = node.hOpsList[-1]
    #         self.evalDict["Node{0}_H".format(node.index)] = h_output
    #     # Activations
    #     for k, v in node.activationsDict.items():
    #         self.evalDict["Node{0}_activation_from_{1}".format(node.index, k)] = v
    #     # Decision masks
    #     for k, v in node.maskTensors.items():
    #         self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
    #     # Evaluation outputs
    #     for k, v in node.evalDict.items():
    #         self.evalDict[k] = v
    #     # Label outputs
    #     if node.labelTensor is not None:
    #         self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
    #         # Sample indices
    #         self.evalDict["Node{0}_indices_tensor".format(node.index)] = node.indicesTensor
    #     # One Hot Label outputs
    #     if node.oneHotLabelTensor is not None:
    #         self.evalDict["Node{0}_one_hot_label_tensor".format(node.index)] = node.oneHotLabelTensor
    #     if node.filteredMask is not None:
    #         self.evalDict["Node{0}_filteredMask".format(node.index)] = node.filteredMask
    #
    # self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
    # self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
    # self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}

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
        ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
        node_degree = self.degreeList[node.depth]
        # MultiGPU OK
        hyperplane_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
            shape=[ig_feature_size, node_degree],
            type=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
                [ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        # MultiGPU OK
        hyperplane_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
            shape=[node_degree],
            type=GlobalConstants.DATA_TYPE,
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
            mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            node.maskTensors[child_index] = mask_tensor
            node.masksWithoutThreshold[child_index] = mask_without_threshold
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors
            node.evalDict[self.get_variable_name(name="masksWithoutThreshold", node=node)] = node.masksWithoutThreshold
