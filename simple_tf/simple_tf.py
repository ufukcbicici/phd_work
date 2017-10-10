from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from auxillary.constants import DatasetTypes
from auxillary.dag_utilities import Dag

import argparse
import gzip
import os
import sys
import time
import networkx as nx

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

# MNIST
from data_handling.mnist_data_set import MnistDataSet

EPOCH_COUNT = 1000
BATCH_SIZE = 1000
EVAL_BATCH_SIZE = 50000
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NO_FILTERS_1 = 20
NO_FILTERS_2 = 16
NO_HIDDEN = 55
NUM_LABELS = 10
WEIGHT_DECAY_COEFFICIENT = 0.0
INITIAL_LR = 0.01
DECAY_STEP = 10000
DECAY_RATE = 0.5
TREE_DEGREE = 3
DATA_TYPE = tf.float32
SEED = None
USE_CPU = True
USE_CPU_MASKING = False
USE_RANDOM_PARAMETERS = True
TRAIN_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
TRAIN_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
TEST_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
TEST_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(EVAL_BATCH_SIZE,))
tf.set_random_seed(1234)
np_seed = 88
np.random.seed(np_seed)


class Node:
    def __init__(self, index, depth, is_root, is_leaf):
        self.index = index
        self.depth = depth
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.variablesList = []
        self.fOpsList = []
        self.hOpsList = []
        self.lossList = []
        self.labelTensor = None
        self.maskTensorsDict = {}
        self.evalDict = {}
        # Indexed by the nodes producing them
        self.activationsDict = {}


class TreeNetwork:
    def __init__(self, tree_degree, node_build_funcs, create_new_variables, data, label):
        self.treeDegree = tree_degree
        self.dagObject = Dag()
        self.nodeBuildFuncs = node_build_funcs
        self.depth = len(self.nodeBuildFuncs)
        self.nodes = {}
        self.topologicalSortedNodes = []
        self.createNewVariables = create_new_variables
        self.dataTensor = data
        self.labelTensor = label
        self.evalDict = {}
        self.finalLoss = None

    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def build_network(self, network_to_copy_from):
        curr_index = 0
        for depth in range(0, self.depth):
            node_count_in_depth = pow(self.treeDegree, depth)
            for i in range(0, node_count_in_depth):
                is_root = depth == 0
                is_leaf = depth == (self.depth - 1)
                node = Node(index=curr_index, depth=depth, is_root=is_root, is_leaf=is_leaf)
                self.nodes[curr_index] = node
                if not is_root:
                    parent_index = self.get_parent_index(node_index=curr_index)
                    self.dagObject.add_edge(parent=self.nodes[parent_index], child=node)
                else:
                    self.dagObject.add_node(node=node)
                curr_index += 1
        # Build symbolic networks
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        for node in self.topologicalSortedNodes:
            if self.createNewVariables:
                self.nodeBuildFuncs[node.depth](node=node, network=self)
            else:
                self.nodeBuildFuncs[node.depth](node=node, network=self,
                                                variables=network_to_copy_from.nodes[node.index].variablesList)
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
            for k, v in node.maskTensorsDict.items():
                self.evalDict["Node{0}_{1}".format(node.index, v.name)] = v
            # Evaluation outputs
            for k, v in node.evalDict.items():
                self.evalDict[k] = v
            # Label outputs
            if node.labelTensor is not None:
                self.evalDict["Node{0}_label_tensor".format(node.index)] = node.labelTensor
        # Prepare losses
        all_network_losses = []
        for node in self.topologicalSortedNodes:
            all_network_losses.extend(node.lossList)
        # Weight decays
        vars = tf.trainable_variables()
        weights_and_filters = [v for v in vars if "bias" not in v.name]
        regularizer_loss = WEIGHT_DECAY_COEFFICIENT * tf.add_n([tf.nn.l2_loss(v) for v in weights_and_filters])
        actual_loss = tf.add_n(all_network_losses)
        self.finalLoss = actual_loss + regularizer_loss
        self.evalDict["RegularizerLoss"] = regularizer_loss
        self.evalDict["ActualLoss"] = actual_loss
        self.evalDict["NetworkLoss"] = self.finalLoss

    def calculate_accuracy(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        while True:
            results = eval_network(sess=sess, network=self, dataset=dataset)
            for node in self.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                posterior_probs = results[get_variable_name(name="posterior_probs", node=node)]
                true_labels = results[get_variable_name(name="labels", node=node)]
                predicted_labels = np.argmax(posterior_probs, axis=1)
                if node.index not in leaf_predicted_labels_dict:
                    leaf_predicted_labels_dict[node.index] = predicted_labels
                else:
                    leaf_predicted_labels_dict[node.index] = np.concatenate((leaf_predicted_labels_dict[node.index],
                                                                             predicted_labels))
                if node.index not in leaf_true_labels_dict:
                    leaf_true_labels_dict[node.index] = true_labels
                else:
                    leaf_true_labels_dict[node.index] = np.concatenate((leaf_true_labels_dict[node.index], true_labels))
            if dataset.isNewEpoch:
                break
        print("****************Dataset:{0}****************".format(dataset_type))
        overall_count = 0.0
        overall_correct = 0.0
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            predicted = leaf_predicted_labels_dict[node.index]
            true_labels = leaf_true_labels_dict[node.index]
            if predicted.shape != true_labels.shape:
                raise Exception("Predicted and true labels counts do not hold.")
            correct_count = np.sum(predicted == true_labels)
            total_count = true_labels.shape[0]
            overall_correct += correct_count
            overall_count += total_count
            if total_count > 0:
                print("Leaf {0}: Sample Count:{1} Accuracy:{2}".format(node.index, total_count,
                                                                       float(correct_count) / float(total_count)))
            else:
                print("Leaf {0} is empty.".format(node.index))
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(overall_count, overall_correct / overall_count))


def get_variable_name(name, node):
    return "Node{0}_{1}".format(node.index, name)


def apply_decision(node, network):
    # child_nodes = sorted(network.dagObject.children(node=node), key=lambda child: child.index)
    arg_max_indices = tf.argmax(input=node.activationsDict[node.index], axis=1)
    node.maskTensorsDict = {}
    for index in range(network.treeDegree):
        child_index = node.index * network.treeDegree + 1 + index
        mask_vector = tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64), name="Mask_{0}".format(child_index))
        mask_vector = tf.reshape(mask_vector, [-1])
        node.maskTensorsDict[node.index * network.treeDegree + 1 + index] = mask_vector


def baseline(node, network, variables=None):
    # Parameters
    if network.createNewVariables:
        conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, NO_FILTERS_1], stddev=0.1, seed=SEED,
                                                        dtype=DATA_TYPE), name=get_variable_name(name="conv1_weight",
                                                                                                 node=node))
        conv1_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_1], dtype=DATA_TYPE),
                                   name=get_variable_name(name="conv1_bias",
                                                          node=node))
        conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, NO_FILTERS_1, NO_FILTERS_2], stddev=0.1, seed=SEED, dtype=DATA_TYPE),
            name=get_variable_name(name="conv2_weight", node=node))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_2], dtype=DATA_TYPE),
                                   name=get_variable_name(name="conv2_bias",
                                                          node=node))
        fc_weights_1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * NO_FILTERS_2, NO_HIDDEN],
                                                       stddev=0.1, seed=SEED, dtype=DATA_TYPE),
                                   name=get_variable_name(name="fc_weights_1", node=node))
        fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[NO_HIDDEN], dtype=DATA_TYPE),
                                  name=get_variable_name(name="fc_biases_1", node=node))
        fc_weights_2 = tf.Variable(tf.truncated_normal([NO_HIDDEN + network.depth - 1, NUM_LABELS],
                                                       stddev=0.1,
                                                       seed=SEED,
                                                       dtype=DATA_TYPE),
                                   name=get_variable_name(name="fc_weights_2", node=node))
        fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=DATA_TYPE),
                                  name=get_variable_name(name="fc_biases_2", node=node))
        node.variablesList.extend(
            [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc_weights_1, fc_biases_1, fc_weights_2,
             fc_biases_2])
    else:
        node.variablesList = []
        node.variablesList.extend(variables)
        conv1_weights = node.variablesList[0]
        conv1_biases = node.variablesList[1]
        conv2_weights = node.variablesList[2]
        conv2_biases = node.variablesList[3]
        fc_weights_1 = node.variablesList[4]
        fc_biases_1 = node.variablesList[5]
        fc_weights_2 = node.variablesList[6]
        fc_biases_2 = node.variablesList[7]
    # Operations
    conv1 = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flattened = tf.contrib.layers.flatten(pool2)
    hidden_1 = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    logits = tf.matmul(hidden_1, fc_weights_2) + fc_biases_2
    # Loss
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=network.labelTensor,
                                                                               logits=logits)
    loss = tf.reduce_mean(cross_entropy_loss_tensor)
    node.fOpsList.extend([conv1, relu1, pool1, conv2, relu2, pool2, flattened, hidden_1, logits,
                          cross_entropy_loss_tensor, loss])
    node.lossList.append(loss)
    # Evaluation
    node.evalDict[get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[get_variable_name(name="labels", node=node)] = network.labelTensor


def root_func(node, network, variables=None):
    # Parameters
    if USE_RANDOM_PARAMETERS:
        conv_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, NO_FILTERS_1], stddev=0.1, seed=SEED,
                                                       dtype=DATA_TYPE), name=get_variable_name(name="conv_weight",
                                                                                                node=node))
    else:
        conv_weights = tf.Variable(tf.constant(0.1, shape=[5, 5, NUM_CHANNELS, NO_FILTERS_1], dtype=DATA_TYPE),
                                   name=get_variable_name(name="conv_weight", node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_1], dtype=DATA_TYPE),
                              name=get_variable_name(name="conv_bias",
                                                     node=node))
    if USE_RANDOM_PARAMETERS:
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, network.treeDegree], stddev=0.1, seed=SEED, dtype=DATA_TYPE),
            name=get_variable_name(name="hyperplane_weights", node=node))
    else:
        hyperplane_weights = tf.Variable(
            tf.constant(value=0.1, shape=[IMAGE_SIZE * IMAGE_SIZE, network.treeDegree]),
            name=get_variable_name(name="hyperplane_weights", node=node))

    # print_op = tf.Print(input_=network.dataTensor, data=[network.dataTensor], message="Print at Node:{0}".format(node.index))
    # node.evalDict[get_variable_name(name="Print", node=node)] = print_op
    hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[network.treeDegree], dtype=DATA_TYPE),
                                    name=get_variable_name(name="hyperplane_biases", node=node))
    node.variablesList.extend([conv_weights, conv_biases, hyperplane_weights, hyperplane_biases])
    # Operations
    # F
    conv = tf.nn.conv2d(network.dataTensor, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    flat_data = tf.contrib.layers.flatten(network.dataTensor)
    node.hOpsList.extend([flat_data])
    # Decisions
    node.activationsDict[node.index] = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
    apply_decision(node=node, network=network)


def l1_func(node, network, variables=None):
    # Parameters
    if USE_RANDOM_PARAMETERS:
        conv_weights = tf.Variable(
            tf.truncated_normal([5, 5, NO_FILTERS_1, NO_FILTERS_2], stddev=0.1, seed=SEED, dtype=DATA_TYPE),
            name=get_variable_name(name="conv_weight", node=node))
    else:
        conv_weights = tf.Variable(
            tf.constant(0.1, shape=[5, 5, NO_FILTERS_1, NO_FILTERS_2], dtype=DATA_TYPE),
            name=get_variable_name(name="conv_weight", node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_2], dtype=DATA_TYPE),
                              name=get_variable_name(name="conv_bias",
                                                     node=node))
    if USE_RANDOM_PARAMETERS:
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE + network.treeDegree, network.treeDegree], stddev=0.1, seed=SEED, dtype=DATA_TYPE),
            name=get_variable_name(name="hyperplane_weights", node=node))
    else:
        hyperplane_weights = tf.Variable(
            tf.constant(0.1, shape=[IMAGE_SIZE * IMAGE_SIZE + network.treeDegree, network.treeDegree], dtype=DATA_TYPE),
            name=get_variable_name(name="hyperplane_weights", node=node))
    hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[network.treeDegree], dtype=DATA_TYPE),
                                    name=get_variable_name(name="hyperplane_biases", node=node))
    node.variablesList.extend([conv_weights, conv_biases, hyperplane_weights, hyperplane_biases])
    # Operations
    # Mask inputs
    parent_node = network.dagObject.parents(node=node)[0]
    # print_op = tf.Print(input_=parent_node.fOpsList[-1], data=[parent_node.fOpsList[-1]], message="Print at Node:{0}".format(node.index))
    # node.evalDict[get_variable_name(name="Print", node=node)] = print_op
    mask_tensor = parent_node.maskTensorsDict[node.index]
    if USE_CPU_MASKING:
        with tf.device("/cpu:0"):
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(network.labelTensor, mask_tensor)
    else:
        parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
        parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
        for k, v in parent_node.activationsDict.items():
            node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
        node.labelTensor = tf.boolean_mask(network.labelTensor, mask_tensor)
    # F
    conv = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    node.hOpsList.extend([parent_H])
    # Decisions
    concat_list = [parent_H]
    concat_list.extend(node.activationsDict.values())
    h_concat = tf.concat(values=concat_list, axis=1)
    node.activationsDict[node.index] = tf.matmul(h_concat, hyperplane_weights) + hyperplane_biases
    apply_decision(node=node, network=network)


def leaf_func(node, network, variables=None):
    # Parameters
    if USE_RANDOM_PARAMETERS:
        fc_weights_1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * NO_FILTERS_2, NO_HIDDEN],
                                                       stddev=0.1, seed=SEED, dtype=DATA_TYPE),
                                   name=get_variable_name(name="fc_weights_1", node=node))
        fc_weights_2 = tf.Variable(tf.truncated_normal([NO_HIDDEN + 2 * network.treeDegree, NUM_LABELS],
                                                       stddev=0.1,
                                                       seed=SEED,
                                                       dtype=DATA_TYPE),
                                   name=get_variable_name(name="fc_weights_2", node=node))
    else:
        fc_weights_1 = tf.Variable(tf.constant(0.1, shape=[IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * NO_FILTERS_2, NO_HIDDEN], dtype=DATA_TYPE),
                                   name=get_variable_name(name="fc_weights_1", node=node))
        fc_weights_2 = tf.Variable(tf.constant(0.1, shape=[NO_HIDDEN + 2 * network.treeDegree, NUM_LABELS],
                                                       dtype=DATA_TYPE),
                                   name=get_variable_name(name="fc_weights_2", node=node))
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[NO_HIDDEN], dtype=DATA_TYPE),
                              name=get_variable_name(name="fc_biases_1", node=node))
    fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=DATA_TYPE),
                              name=get_variable_name(name="fc_biases_2", node=node))
    node.variablesList.extend([fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2])
    # Operations
    # Mask inputs
    parent_node = network.dagObject.parents(node=node)[0]
    # print_op = tf.Print(input_=parent_node.fOpsList[-1], data=[parent_node.fOpsList[-1]], message="Print at Node:{0}".format(node.index))
    # node.evalDict[get_variable_name(name="Print", node=node)] = print_op
    mask_tensor = parent_node.maskTensorsDict[node.index]
    if USE_CPU_MASKING:
        with tf.device("/cpu:0"):
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            # parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
    else:
        parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
        # parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
        for k, v in parent_node.activationsDict.items():
            node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
        node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
    # Loss
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    # Concatenate activations from ancestors
    concat_list = [hidden_layer]
    concat_list.extend(node.activationsDict.values())
    hidden_layer_concat = tf.concat(values=concat_list, axis=1)
    logits = tf.matmul(hidden_layer_concat, fc_weights_2) + fc_biases_2
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                               logits=logits)
    pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
    loss = tf.where(tf.is_nan(pre_loss), 0., pre_loss)
    node.fOpsList.extend([flattened, hidden_layer, hidden_layer_concat, logits, cross_entropy_loss_tensor, pre_loss,
                          loss])
    node.lossList.append(loss)
    # Evaluation
    node.evalDict[get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[get_variable_name(name="labels", node=node)] = node.labelTensor
    node.evalDict[get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)


def eval_network(sess, network, dataset):
    # if is_train:
    samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
    samples = np.expand_dims(samples, axis=3)
    feed_dict = {TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels}
    results = sess.run(network.evalDict, feed_dict)
    # else:
    #     samples, labels, indices_list = dataset.get_next_batch(batch_size=EVAL_BATCH_SIZE)
    #     samples = np.expand_dims(samples, axis=3)
    #     feed_dict = {TEST_DATA_TENSOR: samples, TEST_LABEL_TENSOR: labels}
    #     results = sess.run(network.evalDict, feed_dict)
    return results


def main():
    # Build the network
    # network = TreeNetwork(tree_degree=2, node_build_funcs=[baseline], create_new_variables=True,
    #                                data=TRAIN_DATA_TENSOR, label=TRAIN_LABEL_TENSOR)
    # network.build_network(network_to_copy_from=None)
    network = TreeNetwork(tree_degree=TREE_DEGREE, node_build_funcs=[root_func, l1_func, leaf_func], create_new_variables=True,
                          data=TRAIN_DATA_TENSOR, label=TRAIN_LABEL_TENSOR)
    network.build_network(network_to_copy_from=None)
    # test_network = TreeNetwork(tree_degree=2, node_build_funcs=[baseline], create_new_variables=False,
    #                            data=TEST_DATA_TENSOR, label=TEST_LABEL_TENSOR)
    # test_network.build_network(network_to_copy_from=training_network)
    # Do the training
    if USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    dataset = MnistDataSet(validation_sample_count=10000)
    # Acquire the losses for training
    loss_list = []
    vars = tf.trainable_variables()
    var_names = [v.name for v in vars]
    # Train
    # Setting the optimizer
    # global_counter = tf.Variable(0, dtype=DATA_TYPE, trainable=False)
    # learning_rate = tf.train.exponential_decay(
    #     INITIAL_LR,  # Base learning rate.
    #     global_counter,  # Current index into the dataset.
    #     DECAY_STEP,  # Decay step.
    #     DECAY_RATE,  # Decay rate.
    #     staircase=True)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(network.finalLoss, global_step=global_counter)
    gradients = tf.gradients(ys=network.finalLoss, xs=vars)
    # Init variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Experiment
    samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
    samples = np.expand_dims(samples, axis=3)
    # eval
    # params = {v.name: v for node in network.topologicalSortedNodes for v in node.variablesList}
    # param_values = sess.run(params, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    # node3_eval = {k: v for k, v in network.evalDict.items() if "Node3" in k}
    # node4_eval = {k: v for k, v in network.evalDict.items() if "Node4" in k}
    # node5_eval = {k: v for k, v in network.evalDict.items() if "Node5" in k}
    # node6_eval = {k: v for k, v in network.evalDict.items() if "Node6" in k}

    # for run_id in range(100):
    #     # results = sess.run(network.evalDict, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     results3 = sess.run(node3_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node3")
    # for run_id in range(100):
    #     results4 = sess.run(node4_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node4")
    # for run_id in range(100):
    #     results5 = sess.run(node5_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node5")
    # for run_id in range(100):
    #     results6 = sess.run(node6_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node6")
    #
    # for run_id in range(100):
    #     results6 = sess.run(node6_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node6")
    # for run_id in range(100):
    #     start_time = time.time()
    #     results = sess.run(network.evalDict, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     elapsed_time = time.time() - start_time
    #     print("elapsed time={0}".format(elapsed_time))
    #     for k, v in results.items():
    #         if "sample_count" in k:
    #             print("{0}={1}".format(k, v))
    #     print("X")

    # First loss
    network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training)
    network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation)
    lr = 9000.0
    sample_count_tensors = {k: network.evalDict[k] for k in network.evalDict.keys() if "sample_count" in k}
    for epoch_id in range(EPOCH_COUNT):
        # An epoch is a complete pass on the whole dataset.
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
        print("*************Epoch {0}*************".format(epoch_id))
        total_time = 0.0
        while True:
            samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
            samples = np.expand_dims(samples, axis=3)
            start_time = time.time()
            feed_dict = {TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels}
            results = sess.run([gradients, sample_count_tensors], feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            # print("Iteration:{0}".format(results[1]))
            print("Iteration:{0} Elapsed Time:{1}".format(results[1], elapsed_time))
            # Print sample counts
            for k, v in results[3].items():
                print("{0}={1}".format(k, v))
            # print("Iteration:{0}".format(results[1]))
            if abs(results[0] - lr) > 1e-10:
                print("Learning rate changed to {0}".format(results[0]))
                lr = results[0]
            # print("lr={0}".format(results[0]))
            # print("global counter={0}".format(results[1]))
            if dataset.isNewEpoch:
                print("Epoch Time={0}".format(total_time))
                network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training)
                network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation)
                break
    print("X")


def experiment():
    sess = tf.Session()
    dataset = MnistDataSet(validation_sample_count=10000)
    conv_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, NO_FILTERS_1], stddev=0.1, seed=SEED,
                                                   dtype=DATA_TYPE), name="conv_weight")
    conv_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_1], dtype=DATA_TYPE), name="conv_bias")
    hyperplane_weights = tf.Variable(tf.constant(value=0.1, shape=[IMAGE_SIZE * IMAGE_SIZE, TREE_DEGREE]),
                                     name="hyperplane_weights")
    hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[TREE_DEGREE], dtype=DATA_TYPE), name="hyperplane_biases")
    conv = tf.nn.conv2d(TRAIN_DATA_TENSOR, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flat_data = tf.contrib.layers.flatten(TRAIN_DATA_TENSOR)
    activations = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
    arg_max_indices = tf.argmax(input=activations, axis=1)
    maskTensorsDict = {}
    outputsDict = {}
    sumDict = {}
    secondMasks = {}
    secondMaskOutput = {}
    activationsDict = {}
    for index in range(TREE_DEGREE):
        mask_vector = tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64), name="Mask_{0}".format(index))
        maskTensorsDict[index] = mask_vector
        outputsDict[index] = tf.boolean_mask(pool, mask_vector)
        activationsDict[index] = tf.boolean_mask(activations, mask_vector)
        flattened = tf.contrib.layers.flatten(outputsDict[index])
        sum_vec = tf.reduce_sum(flattened, axis=1)
        sumDict[index] = sum_vec
        second_mask = tf.greater_equal(x=sum_vec, y=tf.constant(550.0, tf.float32))
        secondMasks[index] = second_mask
        secondMaskOutput[index] = tf.boolean_mask(activationsDict[index], second_mask)
    init = tf.global_variables_initializer()
    sess.run(init)
    samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
    samples = np.expand_dims(samples, axis=3)
    for run_id in range(100):
        feed_dict = {TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels}
        results = sess.run([maskTensorsDict, outputsDict, activationsDict, arg_max_indices, sumDict, secondMaskOutput,
                            secondMasks],
                           feed_dict=feed_dict)
        print("X")



main()
# experiment()
