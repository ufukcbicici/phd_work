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

# MNIST
from data_handling.mnist_data_set import MnistDataSet

EPOCH_COUNT = 100
BATCH_SIZE = 1000
EVAL_BATCH_SIZE = 50000
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NO_FILTERS_1 = 20
NO_FILTERS_2 = 50
NO_HIDDEN = 500
NUM_LABELS = 10
DATA_TYPE = tf.float32
SEED = None
TRAIN_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
TRAIN_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
TEST_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
TEST_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(EVAL_BATCH_SIZE,))


class Node:
    def __init__(self, index, depth, is_root, is_leaf):
        self.index = index
        self.depth = depth
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.variablesList = []
        self.fOpsList = []
        self.hOpsList = []
        self.activationsTensor = None
        self.decisionsDict = {}


class TreeNetwork:
    def __init__(self, tree_degree, node_build_funcs):
        self.treeDegree = tree_degree
        self.dagObject = Dag()
        self.nodeBuildFuncs = node_build_funcs
        self.depth = len(self.nodeBuildFuncs)
        self.nodes = {}
        self.topologicalSortedNodes = []

    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def build_network(self):
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
            self.nodeBuildFuncs[node.depth](node=node, network=self)


def get_variable_name(name, node):
    return "node{0}_{1}".format(node.index, name)


def apply_decision(node, network):
    # child_nodes = sorted(network.dagObject.children(node=node), key=lambda child: child.index)
    arg_max_indices = tf.argmax(input=node.activationsTensor, axis=1)
    node.decisionsDict = {}
    for index in range(network.treeDegree):
        mask_vector = tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64))
        node.decisionsDict[node.index * network.treeDegree + 1 + index] = mask_vector


def root_func(node, network):
    # Parameters
    conv_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, NO_FILTERS_1], stddev=0.1, seed=SEED,
                                                   dtype=DATA_TYPE), name=get_variable_name(name="conv_weight",
                                                                                            node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_1], dtype=DATA_TYPE),
                              name=get_variable_name(name="conv_bias",
                                                     node=node))
    hyperplane_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, network.treeDegree], stddev=0.1, seed=SEED, dtype=DATA_TYPE),
        name=get_variable_name(name="hyperplane_weights", node=node))
    hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[network.treeDegree], dtype=DATA_TYPE),
                              name=get_variable_name(name="hyperplane_biases", node=node))
    node.variablesList.extend([conv_weights, conv_biases])
    # Operations
    # F
    conv = tf.nn.conv2d(TRAIN_DATA_TENSOR, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    flat_data = tf.contrib.layers.flatten(TRAIN_DATA_TENSOR)
    node.hOpsList.extend([flat_data])
    # Decisions
    node.activationsTensor = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
    apply_decision(node=node,  network=network)


# def l1_func(node, network):
#     # Parameters
#     conv_weights = tf.Variable(
#         tf.truncated_normal([5, 5, NO_FILTERS_1, NO_FILTERS_2], stddev=0.1, seed=SEED, dtype=DATA_TYPE),
#         name=get_variable_name(name="conv_weight", node=node))
#     conv_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_2], dtype=DATA_TYPE),
#                               name=get_variable_name(name="conv_bias",
#                                                      node=node))
#     node.variablesList.extend([conv_weights, conv_biases])
#     # Operations
#     parent_node = network.dagObject.parents(node=node)[0]
#     parent_pool = parent_node.fOpsList[-1]
#     conv = tf.nn.conv2d(parent_pool, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
#     relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
#     pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     node.fOpsList.extend([conv, relu, pool])
#
#
# def leaf_func(node, network):
#     # Parameters
#     fc_weights_1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * NO_FILTERS_2, NO_HIDDEN],
#                                                    stddev=0.1, seed=SEED, dtype=DATA_TYPE),
#                                name=get_variable_name(name="fc_weights_1", node=node))
#     fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[NO_HIDDEN], dtype=DATA_TYPE),
#                               name=get_variable_name(name="fc_biases_1", node=node))
#     fc_weights_2 = tf.Variable(tf.truncated_normal([NO_HIDDEN + network.depth - 1, NUM_LABELS],
#                                                    stddev=0.1,
#                                                    seed=SEED,
#                                                    dtype=DATA_TYPE),
#                                name=get_variable_name(name="fc_weights_2", node=node))
#     fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=DATA_TYPE),
#                               name=get_variable_name(name="fc_biases_2", node=node))
#     node.variablesList.extend([fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2])
#     # Operations
#     parent_node = network.dagObject.parents(node=node)[0]
#     parent_pool = parent_node.fOpsList[-1]
#     flattened = tf.contrib.layers.flatten(parent_pool)
#     hidden_1_r = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1_r)


def main():
    # Build the network
    network = TreeNetwork(tree_degree=2, node_build_funcs=[root_func])
    network.build_network()

    # Do the training
    dataset = MnistDataSet(validation_sample_count=10000)
    with tf.device("/cpu:0"):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=config) as sess:
            # Train
            for epoch_id in range(EPOCH_COUNT):
                # An epoch is a complete pass on the whole dataset.
                dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
                print("*************Epoch {0}*************".format(epoch_id))
                while True:
                    samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
                    feed_dict = {TRAIN_DATA_TENSOR: samples,
                                 TRAIN_LABEL_TENSOR: labels}
                    eval_list = []
                    eval_list.extend(network.nodes[0].fOpsList)
                    eval_list.extend(network.nodes[0].hOpsList)
                    eval_list.append(network.nodes[0].activationsTensor)
                    eval_list.extend(network.nodes[0].decisionsDict.values())
                    evaluationResults = sess.run(eval_list, feed_dict)
                    if dataset.isNewEpoch:
                        break


main()