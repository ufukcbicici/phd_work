from collections import deque
import tensorflow as tf
import numpy as np
from simple_tf.cign.fast_tree import FastTreeNetwork

# USE CLASS BALANCING
from simple_tf.global_params import GlobalConstants
from simple_tf.node import Node


class ClassWeightedTree(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        self.classWeightsDict = {}
        self.leafLabelTensorsDict = {}
        self.leafClassWeightTensorsDict = {}
        self.leafSampleWeightTensorsDict = {}

    def build_network(self):
        # Call super
        super().build_network()
        # Add class weighting data
        for curr_node in self.topologicalSortedNodes:
            if curr_node.isLeaf:
                self.leafClassWeightTensorsDict[curr_node.index] = \
                    tf.placeholder(name="class_weight_tensor_node{0}".format(curr_node.index), dtype=tf.float32)
                assert curr_node.index in self.leafClassWeightTensorsDict \
                       and self.leafClassWeightTensorsDict[curr_node.index] is not None
                assert curr_node.index in self.leafSampleWeightTensorsDict \
                       and self.leafSampleWeightTensorsDict[curr_node.index] is not None
                self.evalDict["Node{0}_leafClassWeightTensor".format(curr_node.index)] = \
                    self.leafClassWeightTensorsDict[curr_node.index]
                self.evalDict["Node{0}_leafSampleWeightTensor".format(curr_node.index)] = \
                    self.leafSampleWeightTensorsDict[curr_node.index]
                self.classWeightsDict[curr_node.index] = np.ones(shape=(self.labelCount,), dtype=np.float32)
                self.leafLabelTensorsDict[curr_node.index] = curr_node.labelTensor

    # Modified Loss Calculation with Class Weighting
    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(
            final_feature)
        logits = tf.matmul(final_feature, softmax_weights) + softmax_biases
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                                   logits=logits)
        self.leafSampleWeightTensorsDict[node.index] = \
            tf.nn.embedding_lookup(self.leafClassWeightTensorsDict[node.index], node.labelTensor)
        cross_entropy_loss_tensor = tf.multiply(self.leafSampleWeightTensorsDict[node.index],
                                                cross_entropy_loss_tensor)
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
        node.lossList.append(loss)
        return final_feature, logits

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking, batch_size):
        feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                              is_train=is_train, use_masking=use_masking, batch_size=batch_size)
        if is_train:
            if not self.isBaseline:
                for node in self.topologicalSortedNodes:
                    if node.isLeaf:
                        feed_dict[self.leafClassWeightTensorsDict[node.index]] = self.classWeightsDict[node.index]
        else:
            if not self.isBaseline:
                for node in self.topologicalSortedNodes:
                    if node.isLeaf:
                        feed_dict[self.leafClassWeightTensorsDict[node.index]] = \
                            np.ones(shape=(self.labelCount,), dtype=np.float32)
        return feed_dict

    def get_run_ops(self):
        run_ops = [self.optimizer, self.learningRate, self.sampleCountTensors, self.isOpenTensors,
                   self.infoGainDicts, self.leafClassWeightTensorsDict, self.leafSampleWeightTensorsDict,
                   self.leafLabelTensorsDict]
        return run_ops

    def calculate_class_weights(self, sample_counts_dict, leaf_labels_dict):
        alpha = GlobalConstants.CLASS_WEIGHT_RUNNING_AVERAGE
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                leaf_sample_count = sample_counts_dict[self.get_variable_name(name="sample_count", node=node)]
                assert leaf_labels_dict[node.index].shape[0] == leaf_sample_count
                for label in range(self.labelCount):
                    label_count = np.sum(leaf_labels_dict[node.index] == label)
                    if label_count == 0:
                        label_count = GlobalConstants.LABEL_EPSILON
                    curr_weight = self.classWeightsDict[node.index][label]
                    new_weight = np.log(leaf_sample_count / float(label_count))
                    self.classWeightsDict[node.index][label] = alpha * curr_weight + (1.0 - alpha) * new_weight

    def update_params(self, sess, dataset, epoch, iteration):
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        minibatch = dataset.get_next_batch()
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
        leaf_label_tensors_dict = results[7]
        self.calculate_class_weights(sample_counts_dict=sample_counts, leaf_labels_dict=leaf_label_tensors_dict)
        # Unit Test for Unified Batch Normalization
        if GlobalConstants.USE_VERBOSE:
            self.verbose_update(eval_dict=results[-1])
        # Unit Test for Unified Batch Normalization
        return lr, sample_counts, is_open_indicators
