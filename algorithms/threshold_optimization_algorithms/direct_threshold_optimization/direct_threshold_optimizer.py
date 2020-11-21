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
        self.sigmoidDecay = None
        self.useHardThreshold = None
        self.branchingLogits = None
        self.temperatures = None
        self.trainIndices, self.testIndices = None, None
        self.routingProbabilities = None
        self.routingProbabilitiesUncalibrated = None
        self.thresholds = None
        self.thresholdsSigmoid = None
        self.hardThresholdTests = {}
        self.softThresholdTests = {}
        self.thresholdTests = {}
        self.pathScores = {}
        self.selectionWeights = None
        self.posteriorsTensor = None
        self.weightedPosteriors = None
        self.weightsArray = None
        self.finalPosteriors = None
        self.predictedLabels = None
        self.accuracy = None

    def threshold_test(self, node, routing_probs):
        # Hard
        self.hardThresholdTests[node.index] = tf.cast(routing_probs >=
                                                      self.thresholdsSigmoid[node.index], tf.float32)
        # Soft
        self.softThresholdTests[node.index] = tf.sigmoid((routing_probs - self.thresholdsSigmoid[node.index]) *
                                                         self.sigmoidDecay)
        thresholds_arr = tf.where(
            self.useHardThreshold, self.hardThresholdTests[node.index], self.softThresholdTests[node.index])
        return thresholds_arr

    def prepare_branching(self):
        for node in self.network.topologicalSortedNodes:
            if node.isRoot:
                parent_node = None
                sibling_index = 0
            else:
                sibling_nodes = self.network.dagObject.siblings(node=node)
                sibling_index = np.argmax(np.array([nd.index == node.index for nd in sibling_nodes]))
                parent_node = self.network.dagObject.parents(node=node)
                assert len(parent_node) == 1
                parent_node = parent_node[0]

            if not node.isLeaf:
                routing_probs = self.routingProbabilities[node.index]
                self.thresholdTests[node.index] = self.threshold_test(node=node, routing_probs=routing_probs)
                if node.isRoot:
                    self.pathScores[node.index] = tf.identity(self.thresholdTests[node.index])
                else:
                    self.pathScores[node.index] = self.thresholdTests[node.index] * tf.expand_dims(
                        self.pathScores[parent_node.index][:, sibling_index], axis=-1)
            else:
                self.pathScores[node.index] = tf.identity(self.pathScores[parent_node.index][:, sibling_index])

    def build_network(self):
        self.posteriorsDict = {node.index: tf.placeholder(dtype=tf.float32,
                                                          name="posteriors_node{0}".format(node.index),
                                                          shape=(None, self.labelCount))
                               for node in self.leafNodes}
        self.gtLabels = tf.placeholder(dtype=tf.int64, shape=(None,), name="gt_labels")
        self.sigmoidDecay = tf.placeholder(dtype=tf.float32, name="sigmoidDecay")
        self.useHardThreshold = tf.placeholder(dtype=tf.bool, name="useHardThreshold")
        self.branchingLogits = {node.index: tf.placeholder(dtype=tf.float32,
                                                           name="brancing_logits_node{0}".format(node.index),
                                                           shape=(None, len(self.network.dagObject.children(node))))
                                for node in self.innerNodes}
        self.temperatures = {node.index: tf.placeholder(dtype=tf.float32,
                                                        name="temperature_node{0}".format(node.index))
                             for node in self.innerNodes}
        self.routingProbabilities = {node.index:
                                         tf.nn.softmax(self.branchingLogits[node.index] / self.temperatures[node.index])
                                     for node in self.innerNodes}
        self.routingProbabilitiesUncalibrated = {node.index: tf.nn.softmax(self.branchingLogits[node.index])
                                                 for node in self.innerNodes}
        # self.thresholds = {node.index: tf.placeholder(dtype=tf.float32,
        #                                               name="thresholds{0}".format(node.index),
        #                                               shape=(1, len(self.network.dagObject.children(node))))
        #                    for node in self.innerNodes}
        self.thresholds = {node.index: tf.get_variable(name="thresholds{0}".format(node.index),
                                                       dtype=tf.float32,
                                                       shape=(1, len(self.network.dagObject.children(node))))
                           for node in self.innerNodes}
        self.thresholdsSigmoid = {node.index: (1.0 / len(self.network.dagObject.children(node))) * tf.sigmoid(
            self.thresholds[node.index]) for node in self.innerNodes}
        self.selectionWeights = None
        # Branching
        self.prepare_branching()

        # Combine all weights
        self.selectionWeights = tf.stack(values=[self.pathScores[node.index] for node in self.leafNodes], axis=-1)
        self.weightsArray = tf.reduce_sum(self.selectionWeights, axis=1, keepdims=True)
        self.weightsArray = tf.reciprocal(self.weightsArray)
        self.selectionWeights = self.selectionWeights * self.weightsArray
        # Combine all posteriors
        self.posteriorsTensor = tf.stack(values=[self.posteriorsDict[node.index] for node in self.leafNodes], axis=-1)
        self.weightedPosteriors = self.posteriorsTensor * tf.expand_dims(self.selectionWeights, axis=1)
        self.finalPosteriors = tf.reduce_sum(self.weightedPosteriors, axis=-1)
        # Performance
        self.predictedLabels = tf.argmax(self.finalPosteriors, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictedLabels, self.gtLabels), tf.float32))

    def calibrate_branching_probabilities(self, run_id, iteration):
        temperatures_dict = {}
        file_name = "network{0}_iteration{1}".format(run_id, iteration)
        if os.path.exists(file_name):
            f = open(file_name, "rb")
            temperatures_dict = pickle.load(f)
            f.close()
        else:
            indices_dict = {self.innerNodes[0].index: self.trainIndices}
            for node in self.innerNodes:
                # Determine clusters, map labels to the most likely children
                node_indices = indices_dict[node.index]
                logits = self.routingData.dictOfDatasets[iteration].get_dict("activations")[node.index][node_indices]
                labels = self.routingData.dictOfDatasets[iteration].labelList[node_indices]
                child_nodes = self.network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda n: n.index)
                selected_branches = np.argmax(logits, axis=-1)
                siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in
                                 enumerate(child_nodes)}
                counters_dict = {}
                for child_node in child_nodes:
                    child_labels = labels[selected_branches == siblings_dict[child_node.index]]
                    label_counter = Counter(child_labels)
                    counters_dict[child_node.index] = label_counter
                    indices_dict[child_node.index] = node_indices[selected_branches == siblings_dict[child_node.index]]
                label_mapping = {}
                for label_id in range(self.labelCount):
                    branch_distribution = [(nd.index, counters_dict[nd.index][label_id])
                                           for nd in child_nodes if label_id in counters_dict[nd.index]]
                    mode_tpl = sorted(branch_distribution, key=lambda tpl: tpl[1], reverse=True)[0]
                    label_mapping[label_id] = siblings_dict[mode_tpl[0]]
                mapped_labels = [label_mapping[l_id] for l_id in labels]
                network_calibration = NetworkCalibrationWithTemperatureScaling(logits=logits, labels=mapped_labels)
                temperature = network_calibration.train()
                temperatures_dict[node.index] = temperature
                tf.reset_default_graph()
            f = open(file_name, "wb")
            pickle.dump(temperatures_dict, f)
            f.close()
        return temperatures_dict

    def train(self, run_id, iteration, test_ratio=0.1):
        indices = np.arange(self.routingData.dictOfDatasets[iteration].labelList.shape[0])
        self.trainIndices, self.testIndices = train_test_split(indices, test_size=test_ratio)
        temperatures_dict = self.calibrate_branching_probabilities(run_id=run_id, iteration=iteration)
        self.build_network()
        sess = tf.Session()

        # thresholds_dict = {}
        # for node in self.innerNodes:
        #     child_count = len(self.network.dagObject.children(node))
        #     thresholds_dict[node.index] = np.random.uniform(low=0.0, high=1.0 / child_count, size=(1, child_count))
        sess.run(tf.global_variables_initializer())
        feed_dict = \
            self.prepare_feed_dict(indices=self.trainIndices,
                                   iteration=iteration,
                                   temperatures_dict=temperatures_dict,
                                   use_hard_threshold=False,
                                   sigmoid_decay=1.0)
        results = sess.run({"accuracy": self.accuracy,
                            "predictedLabels": self.predictedLabels,
                            "gtLabels": self.gtLabels,
                            "finalPosteriors": self.finalPosteriors,
                            "weightsArray": self.weightsArray,
                            "weightedPosteriors": self.weightedPosteriors,
                            "posteriorsTensor": self.posteriorsTensor,
                            "selectionWeights": self.selectionWeights,
                            "pathScores": self.pathScores,
                            "thresholdTests": self.thresholdTests,
                            "softThresholdTests": self.softThresholdTests,
                            "hardThresholdTests": self.hardThresholdTests,
                            "routingProbabilities": self.routingProbabilities,
                            "routingProbabilitiesUncalibrated": self.routingProbabilitiesUncalibrated,
                            "branchingLogits": self.branchingLogits,
                            "thresholds": self.thresholds,
                            "thresholdsSigmoid": self.thresholdsSigmoid},
                           feed_dict=feed_dict)
        print("X")

    def prepare_feed_dict(self, indices, iteration, temperatures_dict,
                          use_hard_threshold, sigmoid_decay):
        feed_dict = {}
        routing_obj = self.routingData.dictOfDatasets[iteration]
        feed_dict[self.gtLabels] = routing_obj.labelList[indices]
        feed_dict[self.useHardThreshold] = use_hard_threshold
        feed_dict[self.sigmoidDecay] = sigmoid_decay
        # Leaf nodes
        for node in self.leafNodes:
            arr = routing_obj.get_dict("posterior_probs")[node.index][indices]
            feed_dict[self.posteriorsDict[node.index]] = arr
        # Inner nodes
        for node in self.innerNodes:
            logits_arr = routing_obj.get_dict("activations")[node.index][indices]
            temperature = temperatures_dict[node.index]
            feed_dict[self.branchingLogits[node.index]] = logits_arr
            feed_dict[self.temperatures[node.index]] = temperature
        return feed_dict
