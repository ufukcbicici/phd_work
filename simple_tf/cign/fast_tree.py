from collections import deque

import numpy as np
import tensorflow as tf
import time

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import FixedParameter, DecayingParameter, DiscreteParameter
from simple_tf.cign.tree import TreeNetwork
from simple_tf.global_params import GlobalConstants, AccuracyCalcType
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node
from auxillary.constants import DatasetTypes


class FastTreeNetwork(TreeNetwork):
    class UpdateResults:
        def __init__(self, lr, sample_counts, is_open_indicators):
            self.lr = lr
            self.sampleCounts = sample_counts
            self.isOpenIndicators = is_open_indicators

    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset)
        self.learningRate = None
        self.optimizer = None
        self.infoGainDicts = None
        self.extra_update_ops = None

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
            self.nodeBuildFuncs[node.depth](node=node, network=self)
        # Build the residue loss
        # self.build_residue_loss()
        # Record all variables into the variable manager (For backwards compatibility)
        # self.variableManager.get_all_node_variables()
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
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
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
        hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                        name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
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
        masked_branching_feature = tf.boolean_mask(branching_feature, node.filteredMask)
        normed_x = CustomBatchNormAlgorithms.masked_batch_norm(x=branching_feature, masked_x=masked_branching_feature,
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

    # MultiGPU OK
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
            # Mask all inputs: F channel, H  channel, activations from ancestors, labels
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

    # MultiGPU OK
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
        minibatch = dataset.get_next_batch()
        if minibatch is None:
            return None, None, None
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
        # Unit Test for Unified Batch Normalization
        update_results = FastTreeNetwork.UpdateResults(lr=lr, sample_counts=sample_counts,
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
            output_arr = results[self.get_variable_name(name=output_name, node=node)]
            UtilityFuncs.concat_to_np_array_dict(dct=collection[output_name], key=node.index, array=output_arr)

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
            results, _ = self.eval_network(sess=sess, dataset=dataset, use_masking=use_masking)
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
            if dataset.isNewEpoch:
                break
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
                     self.filteredMask: np.ones((GlobalConstants.CURR_BATCH_SIZE,), dtype=bool)}
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

    def set_hyperparameters(self, **kwargs):
        pass
        # GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        # GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        # if not self.isBaseline:
        #     GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
        #     GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
        #     self.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
        #                                                             value=kwargs["decision_keep_probability"])
        #
        #     # Noise Coefficient
        #     self.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
        #                                                         decay=0.0,
        #                                                         decay_period=1,
        #                                                         min_limit=0.0)
        #     # Decision Loss Coefficient
        #     # network.decisionLossCoefficientCalculator = DiscreteParameter(name="decision_loss_coefficient_calculator",
        #     #                                                               value=0.0,
        #     #                                                               schedule=[(12000, 1.0)])
        #     self.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
        #                                                             value=1.0)
        #     for node in self.topologicalSortedNodes:
        #         if node.isLeaf:
        #             continue
        #         # Probability Threshold
        #         node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
        #         initial_value = 1.0 / float(node_degree)
        #         threshold_name = self.get_variable_name(name="prob_threshold_calculator", node=node)
        #         # node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
        #         #                                                  decay_period=70000,
        #         #                                                  min_limit=0.4)
        #         node.probThresholdCalculator = FixedParameter(name=threshold_name, value=initial_value)
        #         # Softmax Decay
        #         decay_name = self.get_variable_name(name="softmax_decay", node=node)
        #         node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
        #                                                         value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
        #                                                         decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
        #                                                         decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
        #                                                         min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)

    def get_explanation_string(self):
        pass

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

    def train(self, sess, dataset, run_id):
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            leaf_info_rows = []
            while True:
                start_time = time.time()
                update_results = self.update_params(sess=sess,
                                                    dataset=dataset,
                                                    epoch=epoch_id,
                                                    iteration=iteration_counter)
                if all([update_results.lr, update_results.sampleCounts, update_results.isOpenIndicators]):
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    self.print_iteration_info(iteration_counter=iteration_counter, update_results=update_results)
                    iteration_counter += 1
                if dataset.isNewEpoch:
                    # moving_results_1 = sess.run(moving_stat_vars)
                    is_evaluation_epoch_at_report_period = \
                        epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
                        and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
                    is_evaluation_epoch_before_ending = \
                        epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
                    if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
                        print("Epoch Time={0}".format(total_time))
                        if not self.modeTracker.isCompressed:
                            training_accuracy, training_confusion = \
                                self.calculate_accuracy(sess=sess, dataset=dataset,
                                                        dataset_type=DatasetTypes.training,
                                                        run_id=run_id, iteration=iteration_counter,
                                                        calculation_type=AccuracyCalcType.regular)
                            validation_accuracy, validation_confusion = \
                                self.calculate_accuracy(sess=sess, dataset=dataset,
                                                        dataset_type=DatasetTypes.test,
                                                        run_id=run_id, iteration=iteration_counter,
                                                        calculation_type=AccuracyCalcType.regular)
                            if not self.isBaseline:
                                validation_accuracy_corrected, validation_marginal_corrected = \
                                    self.calculate_accuracy(sess=sess, dataset=dataset,
                                                            dataset_type=DatasetTypes.test,
                                                            run_id=run_id,
                                                            iteration=iteration_counter,
                                                            calculation_type=
                                                            AccuracyCalcType.route_correction)
                                if is_evaluation_epoch_before_ending:
                                    self.calculate_accuracy(sess=sess, dataset=dataset,
                                                            dataset_type=DatasetTypes.test,
                                                            run_id=run_id,
                                                            iteration=iteration_counter,
                                                            calculation_type=
                                                            AccuracyCalcType.multi_path)
                            else:
                                validation_accuracy_corrected = 0.0
                                validation_marginal_corrected = 0.0
                            DbLogger.write_into_table(
                                rows=[(run_id, iteration_counter, epoch_id, training_accuracy,
                                       validation_accuracy, validation_accuracy_corrected,
                                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
                            # DbLogger.write_into_table(rows=leaf_info_rows, table=DbLogger.leafInfoTable, col_count=4)
                            if GlobalConstants.SAVE_CONFUSION_MATRICES:
                                DbLogger.write_into_table(rows=training_confusion, table=DbLogger.confusionTable,
                                                          col_count=7)
                                DbLogger.write_into_table(rows=validation_confusion, table=DbLogger.confusionTable,
                                                          col_count=7)
                        else:
                            training_accuracy_best_leaf, training_confusion_residue = \
                                self.calculate_accuracy(sess=sess, dataset=dataset,
                                                        dataset_type=DatasetTypes.training,
                                                        run_id=run_id, iteration=iteration_counter,
                                                        calculation_type=AccuracyCalcType.regular)
                            validation_accuracy_best_leaf, validation_confusion_residue = \
                                self.calculate_accuracy(sess=sess, dataset=dataset,
                                                        dataset_type=DatasetTypes.test,
                                                        run_id=run_id, iteration=iteration_counter,
                                                        calculation_type=AccuracyCalcType.regular)
                            DbLogger.write_into_table(rows=[(run_id, iteration_counter, epoch_id,
                                                             training_accuracy_best_leaf,
                                                             validation_accuracy_best_leaf,
                                                             validation_confusion_residue,
                                                             0.0, 0.0, "XXX")], table=DbLogger.logsTable,
                                                      col_count=9)
                        leaf_info_rows = []
                    break
            # # Compress softmax classifiers
            # if GlobalConstants.USE_SOFTMAX_DISTILLATION:
            #     do_compress = network.check_for_compression(dataset=dataset, run_id=experiment_id,
            #                                                 iteration=iteration_counter, epoch=epoch_id)
            #     if do_compress:
            #         print("**********************Compressing the network**********************")
            #         network.softmaxCompresser.compress_network_softmax(sess=sess)
            #         print("**********************Compressing the network**********************")
        # except Exception as e:
        #     print(e)
        #     print("ERROR!!!!")
        # Reset the computation graph
