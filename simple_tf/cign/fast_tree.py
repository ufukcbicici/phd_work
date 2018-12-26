from collections import deque

import numpy as np
import tensorflow as tf

from data_handling.data_set import DataSet
from simple_tf.batch_norm import fast_tree_batch_norm
from simple_tf.cign.tree import TreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node


class FastTreeNetwork(TreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        self.learningRate = None
        self.optimizer = None
        self.infoGainDicts = None
        self.extra_update_ops = None
        self.batchSize = tf.placeholder(tf.int32)

    def build_network(self):
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
            self.nodeBuildFuncs[node.depth](node=node, network=self)
        # Build the residue loss
        self.build_residue_loss()
        # Record all variables into the variable manager (For backwards compatibility)
        # self.variableManager.get_all_node_variables()
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss + self.residueLoss
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
        # Prepare tensors to evaluate
        for node in self.topologicalSortedNodes:
            # if node.isLeaf:
            #     continue
            # F
            f_output = node.fOpsList[-1]
            self.evalDict["Node{0}_F".format(node.index)] = f_output
            # H
            if len(node.hOpsList) > 0:
                h_output = node.hOpsList[-1]
                self.evalDict["Node{0}_H".format(node.index)] = h_output
            # Activations
            for k, v in node.activationsDict.items():
                self.evalDict["Node{0}_activation_from_{1}".format(node.index, k)] = v
            # Decision masks
            for k, v in node.maskTensors.items():
                self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
            # Evaluation outputs
            for k, v in node.evalDict.items():
                self.evalDict[k] = v
            # Label outputs
            if node.labelTensor is not None:
                self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
                # Sample indices
                self.evalDict["Node{0}_indices_tensor".format(node.index)] = node.indicesTensor
            # One Hot Label outputs
            if node.oneHotLabelTensor is not None:
                self.evalDict["Node{0}_one_hot_label_tensor".format(node.index)] = node.oneHotLabelTensor
            if node.filteredMask is not None:
                self.evalDict["Node{0}_filteredMask".format(node.index)] = node.filteredMask

        self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}

    def apply_decision(self, node, branching_feature):
        node_degree = self.degreeList[node.depth]
        ig_feature_size = branching_feature.get_shape().as_list()[-1]
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="hyperplane_weights", node=node))
        hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                        name=self.get_variable_name(name="hyperplane_biases", node=node))
        node.variablesSet.add(hyperplane_weights)
        node.variablesSet.add(hyperplane_biases)
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = tf.layers.batch_normalization(inputs=branching_feature,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain,
                                                                               tf.bool))
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
            mask_tensor = tf.where(self.useThresholding > 0, x=mask_with_threshold, y=mask_without_threshold)
            node.maskTensors[child_index] = mask_tensor
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors

    def apply_decision_with_unified_batch_norm(self, node, branching_feature):
        node_degree = self.degreeList[node.depth]
        ig_feature_size = branching_feature.get_shape().as_list()[-1]
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=self.get_variable_name(name="hyperplane_weights", node=node))
        hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                        name=self.get_variable_name(name="hyperplane_biases", node=node))
        node.variablesSet.add(hyperplane_weights)
        node.variablesSet.add(hyperplane_biases)
        masked_branching_feature = tf.boolean_mask(branching_feature, node.filteredMask)
        normed_x = fast_tree_batch_norm(x=branching_feature, masked_x=masked_branching_feature,
                                        network=self, node=node,
                                        decay=GlobalConstants.BATCH_NORM_DECAY,
                                        iteration=self.iterationHolder,
                                        is_training_phase=self.isTrain)
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
            mask_tensor = tf.where(self.useMasking > 0, mask_tensor,
                                   tf.logical_or(x=tf.constant(value=True, dtype=tf.bool), y=mask_tensor))
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # Mask all inputs: F channel, H channel, activations from ancestors, labels
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            if GlobalConstants.USE_UNIFIED_BATCH_NORM:
                node.filteredMask = tf.boolean_mask(mask_without_threshold, mask_tensor)
            return parent_F, parent_H

    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
        logits = tf.matmul(final_feature, softmax_weights) + softmax_biases
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                                   logits=logits)
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
        node.lossList.append(loss)
        return final_feature, logits

    def get_run_ops(self):
        run_ops = [self.optimizer, self.learningRate, self.sampleCountTensors, self.isOpenTensors,
                   self.infoGainDicts]
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

    def update_params_with_momentum(self, sess, dataset, epoch, iteration):
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        if minibatch is None:
            return None, None, None
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                           is_train=True, use_masking=True, batch_size=GlobalConstants.BATCH_SIZE)
        # Prepare result tensors to collect
        run_ops = self.get_run_ops()
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        results = sess.run(run_ops, feed_dict=feed_dict)
        lr = results[1]
        sample_counts = results[2]
        is_open_indicators = results[3]
        # Unit Test for Unified Batch Normalization
        if GlobalConstants.USE_VERBOSE:
            self.verbose_update(eval_dict=results[-1])
        # Unit Test for Unified Batch Normalization
        return lr, sample_counts, is_open_indicators

    def eval_network(self, sess, dataset, use_masking):
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        if minibatch is None:
            return None, None
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=1000000, use_threshold=False,
                                           is_train=False, use_masking=use_masking,
                                           batch_size=GlobalConstants.EVAL_BATCH_SIZE)

        # n_dict = {k: v for k, v in self.evalDict.items() if
        #           "Node10" not in k and "Node11" not in k and "Node12" not in k and "Node13" not in k}
        # n_dict2 = {k: v for k, v in self.evalDict.items() if
        #            "Node6" not in k and "Node7" not in k and "Node8" not in k and "Node9" not in k and
        #            "Node10" not in k and "Node11" not in k and "Node12" not in k and "Node13" not in k}
        # small_dict = {k: v for k, v in self.evalDict.items() if
        #               "Node0" in k or "Node1" in k or "Node2" in k or "Node3" in k or "Node4" in k}
        small_dict = {}
        for node in self.topologicalSortedNodes:
            if node.index >= 6:
                continue
            for k, v in node.evalDict.items():
                assert k not in small_dict
                small_dict[k] = v
        # results = sess.run(self.evalDict, feed_dict)
        for i in range(10000):
            print("Run {0}".format(i))
            results = sess.run(small_dict, feed_dict)
        # for k, v in results.items():
        #     if "final_feature_mag" in k:
        #         print("{0}={1}".format(k, v))
        return results, minibatch

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking, batch_size):
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
                     self.batchSize: batch_size,
                     self.filteredMask: np.ones((batch_size,), dtype=bool)}
        if is_train:
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_PROB
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
