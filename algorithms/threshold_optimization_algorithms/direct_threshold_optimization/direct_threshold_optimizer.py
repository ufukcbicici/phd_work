import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import Counter

from algorithms.cign_activation_cost_calculator import CignActivationCostCalculator
from algorithms.network_calibration import NetworkCalibrationWithTemperatureScaling


class DirectThresholdOptimizer:
    def __init__(self, network, routing_data, seed):
        self.network = network
        self.seed = seed
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.network.leafNodes)}
        self.labelCount = len(set(routing_data.labelList))
        self.posteriorsDict = None
        self.gtLabels = None
        self.sigmoidDecay = None
        self.useHardThreshold = None
        self.branchingLogits = None
        self.temperatures = None
        self.routingProbabilities = None
        self.routingProbabilitiesUncalibrated = None
        self.thresholds = None
        # self.thresholdsSigmoid = None
        # self.hardThresholdTests = {}
        # self.softThresholdTests = {}
        self.thresholdTests = {}
        self.pathScores = {}
        self.selectionTuples = None
        self.selectionWeights = None
        self.posteriorsTensor = None
        self.weightedPosteriors = None
        self.weightsArray = None
        self.finalPosteriors = None
        self.predictedLabels = None
        self.correctnessVector = None
        self.accuracy = None
        self.oneHotGtLabels = None
        self.diffMatrix = None
        self.diffMatrixSquared = None
        self.meanSquaredDistances = None
        self.meanSquaredLoss = None
        self.totalOptimizer = None
        self.totalGlobalStep = None
        self.powersOfTwoArr = None
        self.activationCodes = None
        self.networkActivationCosts, self.networkActivationCostsDict = \
            CignActivationCostCalculator.calculate_mac_cost(
                network=self.network,
                node_costs=routing_data.get_dict("nodeCosts"))
        self.activationCostsArr = None
        self.meanActivationCost = None
        self.mixingLambda = None
        self.score = None
        self.kind = "probability"

    def get_run_dict(self):
        run_dict = {"accuracy": self.accuracy,
                    "predictedLabels": self.predictedLabels,
                    "gtLabels": self.gtLabels,
                    "finalPosteriors": self.finalPosteriors,
                    "weightsArray": self.weightsArray,
                    "weightedPosteriors": self.weightedPosteriors,
                    "posteriorsTensor": self.posteriorsTensor,
                    "selectionWeights": self.selectionWeights,
                    "pathScores": self.pathScores,
                    "thresholdTests": self.thresholdTests,
                    "routingProbabilities": self.routingProbabilities,
                    "routingProbabilitiesUncalibrated": self.routingProbabilitiesUncalibrated,
                    "branchingLogits": self.branchingLogits,
                    "thresholds": self.thresholds,
                    "powersOfTwoArr": self.powersOfTwoArr,
                    "activationCodes": self.activationCodes,
                    "selectionTuples": self.selectionTuples,
                    "networkActivationCosts": self.networkActivationCosts,
                    "activationCostsArr": self.activationCostsArr,
                    "meanActivationCost": self.meanActivationCost,
                    "correctnessVector": self.correctnessVector,
                    "score": self.score}
        return run_dict

    def threshold_test(self, node, routing_probs):
        # Hard
        child_nodes = self.network.dagObject.children(node)
        self.thresholds[node.index] = tf.placeholder(dtype=tf.float32,
                                                     name="thresholds{0}".format(node.index),
                                                     shape=(1, len(child_nodes)))
        comparison_arr = tf.cast(routing_probs >= self.thresholds[node.index], tf.float32)
        return comparison_arr
        # # Soft
        # self.softThresholdTests[node.index] = tf.sigmoid((routing_probs - self.thresholdsSigmoid[node.index]) *
        #                                                  self.sigmoidDecay)
        # thresholds_arr = tf.where(
        #     self.useHardThreshold, self.hardThresholdTests[node.index], self.softThresholdTests[node.index])
        # return thresholds_arr

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
                               for node in self.network.leafNodes}
        self.gtLabels = tf.placeholder(dtype=tf.int64, shape=(None,), name="gt_labels")
        self.totalGlobalStep = tf.Variable(0, name="total_global_step", trainable=False)
        self.sigmoidDecay = tf.placeholder(dtype=tf.float32, name="sigmoidDecay")
        self.useHardThreshold = tf.placeholder(dtype=tf.bool, name="useHardThreshold")
        self.mixingLambda = tf.placeholder(dtype=tf.float64, name="mixingLambda")
        self.branchingLogits = {node.index: tf.placeholder(dtype=tf.float32,
                                                           name="brancing_logits_node{0}".format(node.index),
                                                           shape=(None, len(self.network.dagObject.children(node))))
                                for node in self.network.innerNodes}
        self.temperatures = {node.index: tf.placeholder(dtype=tf.float32,
                                                        name="temperature_node{0}".format(node.index))
                             for node in self.network.innerNodes}
        self.routingProbabilities = {node.index:
                                         tf.nn.softmax(self.branchingLogits[node.index] / self.temperatures[node.index])
                                     for node in self.network.innerNodes}
        self.routingProbabilitiesUncalibrated = {node.index: tf.nn.softmax(self.branchingLogits[node.index])
                                                 for node in self.network.innerNodes}
        self.thresholds = {}
        self.powersOfTwoArr = tf.reverse(2 ** tf.range(len(self.network.leafNodes)), axis=[0])
        self.networkActivationCosts = tf.constant(self.networkActivationCosts)
        # self.thresholds = {node.index: tf.get_variable(name="thresholds{0}".format(node.index),
        #                                                dtype=tf.float32,
        #                                                shape=(1, len(self.network.dagObject.children(node))))
        #                    for node in self.innerNodes}
        # self.thresholdsSigmoid = {node.index: (1.0 / len(self.network.dagObject.children(node))) * tf.sigmoid(
        #     self.thresholds[node.index]) for node in self.innerNodes}
        self.selectionWeights = None
        # Branching
        self.prepare_branching()

        # Combine all weights
        self.selectionTuples = tf.stack(values=[self.pathScores[node.index] for node in self.network.leafNodes],
                                        axis=-1)
        self.weightsArray = tf.reduce_sum(self.selectionTuples, axis=1, keepdims=True)
        self.weightsArray = tf.reciprocal(self.weightsArray)
        self.selectionWeights = self.selectionTuples * self.weightsArray
        # Combine all posteriors
        self.posteriorsTensor = tf.stack(values=[self.posteriorsDict[node.index] for node in self.network.leafNodes],
                                         axis=-1)
        self.weightedPosteriors = self.posteriorsTensor * tf.expand_dims(self.selectionWeights, axis=1)
        self.finalPosteriors = tf.reduce_sum(self.weightedPosteriors, axis=-1)
        # Performance
        self.predictedLabels = tf.argmax(self.finalPosteriors, axis=-1)
        self.correctnessVector = tf.cast(tf.equal(self.predictedLabels, self.gtLabels), tf.float64)
        self.accuracy = tf.reduce_mean(self.correctnessVector)
        self.activationCodes = tf.cast(self.selectionTuples, tf.int32) * tf.expand_dims(self.powersOfTwoArr, axis=0)
        self.activationCodes = tf.reduce_sum(self.activationCodes, axis=1) - 1
        self.networkActivationCosts = tf.expand_dims(self.networkActivationCosts, axis=-1)
        self.activationCodes = tf.stack([self.activationCodes, tf.zeros_like(self.activationCodes)], axis=1)
        self.activationCostsArr = tf.gather_nd(self.networkActivationCosts, self.activationCodes)
        self.meanActivationCost = tf.reduce_mean(self.activationCostsArr)
        self.score = self.mixingLambda * self.accuracy - (1.0 - self.mixingLambda) * self.meanActivationCost
        # Loss for accuracy metric
        # self.oneHotGtLabels = tf.one_hot(self.gtLabels, self.labelCount)
        # self.diffMatrix = self.finalPosteriors - self.oneHotGtLabels
        # self.diffMatrixSquared = tf.square(self.diffMatrix)
        # self.meanSquaredDistances = tf.reduce_sum(self.diffMatrixSquared, axis=-1)
        # self.meanSquaredLoss = tf.reduce_mean(self.meanSquaredDistances)
        # # Optimization
        # self.totalOptimizer = tf.train.AdamOptimizer().minimize(self.meanSquaredLoss,
        #                                                         global_step=self.totalGlobalStep)

    def prepare_feed_dict(self, routing_data, indices, mixing_lambda, temperatures_dict, thresholds_dict):
        feed_dict = {self.gtLabels: routing_data.labelList[indices], self.mixingLambda: mixing_lambda}
        # Leaf nodes
        for node in self.network.leafNodes:
            arr = routing_data.get_dict("posterior_probs")[node.index][indices]
            feed_dict[self.posteriorsDict[node.index]] = arr
        # Inner nodes
        for node in self.network.innerNodes:
            logits_arr = routing_data.get_dict("activations")[node.index][indices]
            temperature = temperatures_dict[node.index]
            thresholds_arr = thresholds_dict[node.index]
            feed_dict[self.branchingLogits[node.index]] = logits_arr
            feed_dict[self.temperatures[node.index]] = temperature
            feed_dict[self.thresholds[node.index]] = thresholds_arr
        return feed_dict

    def run_threshold_calculator(self, sess, routing_data, indices, mixing_lambda, temperatures_dict, thresholds_dict):
        feed_dict = \
            self.prepare_feed_dict(routing_data=routing_data,
                                   indices=indices,
                                   mixing_lambda=mixing_lambda,
                                   temperatures_dict=temperatures_dict,
                                   thresholds_dict=thresholds_dict)
        run_dict = self.get_run_dict()
        results = sess.run(run_dict, feed_dict=feed_dict)
        return results
