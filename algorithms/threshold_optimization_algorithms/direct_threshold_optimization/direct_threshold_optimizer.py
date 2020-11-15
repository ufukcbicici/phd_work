import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import Counter

from algorithms.network_calibration import NetworkCalibrationWithTemperatureScaling


class DirectThresholdOptimizer:
    def __init__(self, network, routing_data):
        self.network = network
        self.routingData = routing_data
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        self.labelCount = len(set(self.routingData.dictOfDatasets[self.routingData.iterations[0]].labelList))
        self.posteriorsDict = None
        self.gtLabels = None
        self.branchingLogits = None
        self.temperatures = None
        self.trainIndices, self.testIndices = None, None

    def build_network(self):
        self.posteriorsDict = {node.index: tf.placeholder(dtype=tf.float32,
                                                          name="posteriors_node{0}".format(node.index),
                                                          shape=(None, self.labelCount))
                               for node in self.leafNodes}
        self.gtLabels = tf.placeholder(dtype=tf.int32, shape=(None,), name="gt_labels")
        self.branchingLogits = {node.index: tf.placeholder(dtype=tf.float32,
                                                           name="brancing_logits_node{0}".format(node.index),
                                                           shape=(None, len(self.network.dagObject.children(node))))
                                for node in self.innerNodes}
        self.temperatures = {node.index: tf.placeholder(dtype=tf.float32,
                                                        name="temperature_node{0}".format(node.index))
                             for node in self.innerNodes}

    def calibrate_branching_probabilities(self, iteration):
        for node in self.innerNodes:
            # Determine clusters, map labels to the most likely children
            logits = self.routingData.dictOfDatasets[iteration].get_dict("activations")[node.index][self.trainIndices]
            labels = self.routingData.dictOfDatasets[iteration].labelList[self.trainIndices]
            child_nodes = self.network.dagObject.children(node)
            child_nodes = sorted(child_nodes, key=lambda n: n.index)
            selected_branches = np.argmax(logits, axis=-1)
            siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in enumerate(child_nodes)}
            counters_dict = {}
            for child_node in child_nodes:
                child_labels = labels[selected_branches == siblings_dict[child_node.index]]
                label_counter = Counter(child_labels)
                counters_dict[child_node.index] = label_counter
            label_mapping = {}
            for label_id in range(self.labelCount):
                branch_distribution = [(nd.index, counters_dict[nd.index][label_id])
                                       for nd in child_nodes if label_id in counters_dict[nd.index]]
                mode_tpl = sorted(branch_distribution, key=lambda tpl: tpl[1], reverse=True)[0]
                label_mapping[label_id] = siblings_dict[mode_tpl[0]]
            mapped_labels = [label_mapping[l_id] for l_id in labels]
            network_calibration = NetworkCalibrationWithTemperatureScaling(logits=logits, labels=mapped_labels)
            temperature = network_calibration.train()
            print("X")

    def train(self, iteration, test_ratio=0.1):
        indices = np.arange(self.routingData.dictOfDatasets[iteration].labelList.shape[0])
        self.trainIndices, self.testIndices = train_test_split(indices, test_size=test_ratio)
        self.calibrate_branching_probabilities(iteration=iteration)
        print("X")
