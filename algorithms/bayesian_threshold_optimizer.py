import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
from datetime import datetime
import pickle

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

from algorithms.branching_probability_calibration import BranchingProbabilityOptimization
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.bayesian_clusterer import BayesianClusterer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite
from collections import Counter


class BayesianThresholdOptimizer:
    def __init__(self,
                 network, routing_data, session, seed, temperatures_dict,
                 cluster_weights, threshold_kind, mixing_lambda, use_parametric_weights):
        self.network = network
        self.routingData = routing_data
        self.session = session
        self.seed = seed
        self.temperaturesDict = temperatures_dict
        self.clusterWeights = cluster_weights
        self.clusterCount = self.clusterWeights.shape[1]
        self.thresholdKind = threshold_kind
        self.mixingLambda = mixing_lambda
        self.useParametricWeights = use_parametric_weights
        if self.thresholdKind == "entropy":
            self.thresholdOptimizer = DirectThresholdOptimizerEntropy(
                network=self.network, routing_data=self.routingData,
                seed=self.seed, use_parametric_weights=self.useParametricWeights)
        else:
            self.thresholdOptimizer = DirectThresholdOptimizer(
                network=self.network, routing_data=self.routingData,
                seed=self.seed,
                use_parametric_weights=self.useParametricWeights)
        self.thresholdOptimizer.build_network()

    def get_random_thresholds(self):
        list_of_threshold_dicts = []
        for cluster_id in range(self.clusterCount):
            thresholds_dict = {}
            for node in self.network.innerNodes:
                child_nodes = self.network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                child_count = len(child_nodes)
                if self.thresholdKind == "probabiliy":
                    max_bound = 1.0 / child_count
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound, size=(1, child_count))
                elif self.thresholdKind == "entropy":
                    max_bound = -np.log(1.0 / child_count)
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound)
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thresholds_dict)
        return list_of_threshold_dicts

    def calculate_threshold_bounds(self):
        pbounds = {}
        for cluster_id in range(self.clusterCount):
            for node in self.network.innerNodes:
                child_nodes = self.network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if self.thresholdKind == "probability":
                    max_bound = 1.0 / len(child_nodes)
                    for c_nd in child_nodes:
                        pbounds["c_{0}_t_{1}_{2}".format(cluster_id, node.index, c_nd.index)] = (0.0, max_bound)
                elif self.thresholdKind == "entropy":
                    max_bound = -np.log(1.0 / len(child_nodes))
                    pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (0.0, max_bound)
                    # if node.isRoot:
                    #     interval_size = max_bound / cluster_count
                    #     pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (cluster_id*interval_size,
                    #                                                             (cluster_id+1)*interval_size)
                    # else:
                    #     pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (0.0, max_bound)
                else:
                    raise NotImplementedError()
        return pbounds

    def calculate_weight_bounds(self, min_boundary, max_boundary):
        pbounds = {}
        for cluster_id in range(self.clusterCount):
            for node in self.network.leafNodes:
                pbounds["c_{0}_leaf_{1}".format(cluster_id, node.index)] = (min_boundary, max_boundary)
        return pbounds

    def decode_threshold_parameters(self, args_dict):
        list_of_threshold_dicts = []
        for cluster_id in range(self.clusterCount):
            thrs_dict = {}
            for node in self.network.innerNodes:
                child_nodes = self.network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if self.thresholdKind == "probability":
                    thrs_arr = np.array([args_dict["c_{0}_t_{1}_{2}".format(cluster_id, node.index, c_nd.index)]
                                         for c_nd in child_nodes])
                    thrs_dict[node.index] = thrs_arr[np.newaxis, :]
                elif self.thresholdKind == "entropy":
                    thrs_dict[node.index] = args_dict["c_{0}_t_{1}".format(cluster_id, node.index)]
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thrs_dict)
        return list_of_threshold_dicts

    def decode_weight_parameters(self, args_dict):
        list_of_routing_weights = []
        for cluster_id in range(self.clusterCount):
            weight_list = []
            for node in self.network.leafNodes:
                weight = args_dict["c_{0}_leaf_{1}".format(cluster_id, node.index)]
                weight_list.append(weight)
            weight_list = np.array(weight_list)
            list_of_routing_weights.append(weight_list)
        return list_of_routing_weights

    def get_thresholding_results(self,
                                 cluster_weights,
                                 indices,
                                 list_of_threshold_dicts,
                                 temperatures_dict,
                                 mixing_lambda,
                                 list_of_weight_arrays=None):
        # Get thresholding results for all clusters
        correctness_results = []
        activation_costs = []
        score_vectors = []
        accuracies = []
        selection_tuples = []
        posteriors_list = []
        cluster_weights = cluster_weights[indices]
        for cluster_id in range(self.clusterCount):
            thrs_dict = list_of_threshold_dicts[cluster_id]
            weights_array = None
            if self.useParametricWeights:
                assert list_of_weight_arrays is not None
                weights_array = list_of_weight_arrays[cluster_id]
                weights_array = np.repeat(weights_array, repeats=indices.shape[0])
            optimizer_results = self.thresholdOptimizer.run_threshold_calculator(sess=self.session,
                                                                                 routing_data=self.routingData,
                                                                                 indices=indices,
                                                                                 mixing_lambda=mixing_lambda,
                                                                                 temperatures_dict=temperatures_dict,
                                                                                 thresholds_dict=thrs_dict,
                                                                                 routing_weights_array=weights_array)
            correctness_vector = optimizer_results["correctnessVector"]
            activation_costs_vector = optimizer_results["activationCostsArr"]
            accuracies.append(optimizer_results["accuracy"])
            selection_tuples.append(optimizer_results["selectionTuples"])
            posteriors_list.append(optimizer_results["posteriorsTensor"])
            correctness_results.append(correctness_vector)
            activation_costs.append(activation_costs_vector)
            score_vector = mixing_lambda * correctness_vector - (1.0 - mixing_lambda) * activation_costs_vector
            score_vectors.append(score_vector)
        argmax_weights = UtilityFuncs.get_argmax_matrix_from_weight_matrix(cluster_weights)
        selections_tensor = np.stack(selection_tuples, axis=-1)
        weighted_tensors = selections_tensor * np.expand_dims(argmax_weights, axis=1)
        selected_tuples = np.sum(weighted_tensors, axis=-1)

        def get_accumulated_metric(list_of_cluster_results):
            metric_matrix = np.stack(list_of_cluster_results, axis=-1)
            weighted_metric_matrix = cluster_weights * metric_matrix
            weighted_metric_vector = np.sum(weighted_metric_matrix, axis=-1)
            final_metric = np.mean(weighted_metric_vector)
            return final_metric, metric_matrix

        # Scores
        final_score, scores_matrix = get_accumulated_metric(list_of_cluster_results=score_vectors)
        # Accuracies
        final_accuracy, accuracy_matrix = get_accumulated_metric(list_of_cluster_results=correctness_results)
        # Calculation Costs
        final_cost, cost_matrix = get_accumulated_metric(list_of_cluster_results=activation_costs)

        results_dict = {"final_score": final_score, "scores_matrix": scores_matrix,
                        "final_accuracy": final_accuracy, "accuracy_matrix": accuracy_matrix,
                        "final_cost": final_cost, "cost_matrix": cost_matrix,
                        "selections_tensor": selections_tensor,
                        "selected_tuples": selected_tuples}
        return results_dict

    def loss_function(self, **kwargs):
        # Convert Bayesian Optimization space sample into usable thresholds
        list_of_threshold_dicts = self.decode_threshold_parameters(kwargs)
        if self.useParametricWeights:
            list_of_routing_weights = self.decode_weight_parameters(kwargs)
        else:
            list_of_routing_weights = None
        results_dict = {}
        train_indices = self.routingData.trainingIndices
        test_indices = self.routingData.testIndices
        sample_indices = {"train": train_indices, "test": test_indices}
        for data_type in ["train", "test"]:
            results = self.get_thresholding_results(
                cluster_weights=self.clusterWeights,
                indices=sample_indices[data_type],
                list_of_threshold_dicts=list_of_threshold_dicts,
                temperatures_dict=self.temperaturesDict,
                mixing_lambda=self.mixingLambda,
                list_of_weight_arrays=list_of_routing_weights)
            results_dict[data_type] = results
        print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2}".format(
            results_dict["train"]["final_accuracy"],
            results_dict["train"]["final_cost"], results_dict["train"]["final_score"]))
        print("Test Accuracy: {0} Test Computation Load:{1} Test Score:{2}".format(
            results_dict["test"]["final_accuracy"],
            results_dict["test"]["final_cost"], results_dict["test"]["final_score"]))
        return results_dict["train"]["final_score"]

    def optimize(self, weight_bound_min=None, weight_bound_max=None):
        train_indices = self.routingData.trainingIndices
        test_indices = self.routingData.testIndices
        # Learn the standard information gain based accuracies
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=self.network,
                                                                               routing_data=self.routingData,
                                                                               indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=self.network,
                                                                              routing_data=self.routingData,
                                                                              indices=test_indices)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        all_bounds = self.calculate_threshold_bounds()
        if self.useParametricWeights:
            assert weight_bound_min is not None and weight_bound_max is not None
            weight_bounds = self.calculate_weight_bounds(min_boundary=weight_bound_min, max_boundary=weight_bound_max)
            all_bounds.update(weight_bounds)
