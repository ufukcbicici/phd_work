import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

from algorithms.branching_probability_calibration import BranchingProbabilityOptimization
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.bayesian_clusterer import BayesianClusterer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite


class MixedBayesianOptimizer:

    @staticmethod
    def get_random_thresholds(cluster_count, network, kind):
        list_of_threshold_dicts = []
        for cluster_id in range(cluster_count):
            thresholds_dict = {}
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                child_count = len(child_nodes)
                if kind == "probabiliy":
                    max_bound = 1.0 / child_count
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound, size=(1, child_count))
                elif kind == "entropy":
                    max_bound = -np.log(1.0 / child_count)
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound)
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thresholds_dict)
        return list_of_threshold_dicts

    @staticmethod
    def calculate_bounds(cluster_count, network, kind):
        pbounds = {}
        for cluster_id in range(cluster_count):
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if kind == "probability":
                    max_bound = 1.0 / len(child_nodes)
                    for c_nd in child_nodes:
                        pbounds["c_{0}_t_{1}_{2}".format(cluster_id, node.index, c_nd.index)] = (0.0, max_bound)
                elif kind == "entropy":
                    max_bound = -np.log(1.0 / len(child_nodes))
                    pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (0.0, max_bound)
                else:
                    raise NotImplementedError()
        return pbounds

    @staticmethod
    def decode_bayesian_optimization_parameters(args_dict, network, cluster_count, kind):
        list_of_threshold_dicts = []
        for cluster_id in range(cluster_count):
            thrs_dict = {}
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if kind == "probabiliy":
                    thrs_arr = np.array([args_dict["c_{0}_t_{1}_{2}".format(cluster_id, node.index, c_nd.index)]
                                         for c_nd in child_nodes])
                    thrs_dict[node.index] = thrs_arr[np.newaxis, :]
                elif kind == "entropy":
                    thrs_dict[node.index] = args_dict["c_{0}_t_{1}".format(cluster_id, node.index)]
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thrs_dict)
        return list_of_threshold_dicts

    @staticmethod
    def get_thresholding_results(sess, clusterer, cluster_count,
                                 threshold_optimizer, routing_data, indices,
                                 list_of_threshold_dicts, temperatures_dict, mixing_lambda):
        # Get features for training samples
        features = routing_data.get_dict("pre_branch_feature")[0][indices]
        # Get cluster weights for training samples
        cluster_weights = clusterer.get_cluster_scores(sess=sess, features=features)
        # Get thresholding results for all clusters
        correctness_results = []
        activation_costs = []
        score_vectors = []
        accuracies = []
        for cluster_id in range(cluster_count):
            thrs_dict = list_of_threshold_dicts[cluster_id]
            optimizer_results = threshold_optimizer.run_threshold_calculator(sess=sess,
                                                                             indices=indices,
                                                                             mixing_lambda=mixing_lambda,
                                                                             temperatures_dict=temperatures_dict,
                                                                             thresholds_dict=thrs_dict)
            correctness_vector = optimizer_results["correctnessVector"]
            activation_costs_vector = optimizer_results["activationCostsArr"]
            accuracies.append(optimizer_results["accuracy"])
            correctness_results.append(correctness_vector)
            activation_costs.append(activation_costs_vector)
            score_vector = mixing_lambda * correctness_vector + (1.0 - mixing_lambda) * activation_costs_vector
            score_vectors.append(score_vector)

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
                        "final_cost": final_cost, "cost_matrix": cost_matrix}
        return results_dict

    @staticmethod
    def optimize(optimization_iterations_count, iteration,
                 cluster_count, fc_layers, run_id, network, routing_data, seed, test_ratio):
        indices = np.arange(routing_data.labelList.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=test_ratio)
        # Learn the standard information gain based accuracies
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                               routing_data=routing_data,
                                                                               indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=network,
                                                                              routing_data=routing_data,
                                                                              indices=test_indices)
        print("train_ig_accuracy={0}".format(train_ig_accuracy))
        print("test_ig_accuracy={0}".format(test_ig_accuracy))
        # Threshold Optimizer
        dto = DirectThresholdOptimizerEntropy(network=network, routing_data=routing_data, seed=seed,
                                              train_indices=train_indices, test_indices=test_indices)
        temperatures_dict = BranchingProbabilityOptimization.calibrate_branching_probabilities(
            network=network, routing_data=routing_data, run_id=run_id, iteration=iteration, indices=train_indices,
            seed=seed)
        dto.build_network()
        # Clusterer
        bc = BayesianClusterer(network=network, routing_data=routing_data, cluster_count=cluster_count,
                               fc_layers=fc_layers)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        mixing_lambda = 1.0

        # Loss Function
        def f_(**kwargs):
            # Convert Bayesian Optimization space sample into usable thresholds
            list_of_threshold_dicts = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                args_dict=kwargs, network=network, cluster_count=cluster_count, kind=dto.kind)
            results_dict = {}
            sample_indices = {"train": train_indices, "test": test_indices}
            for data_type in ["train", "test"]:
                results = MixedBayesianOptimizer.get_thresholding_results(
                    sess=sess,
                    clusterer=bc,
                    cluster_count=cluster_count,
                    threshold_optimizer=dto,
                    routing_data=routing_data,
                    indices=sample_indices[data_type],
                    list_of_threshold_dicts=list_of_threshold_dicts,
                    temperatures_dict=temperatures_dict,
                    mixing_lambda=mixing_lambda)
                results_dict[data_type] = results
            print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2}".format(results_dict["train"]))

        pbounds = MixedBayesianOptimizer.calculate_bounds(cluster_count=cluster_count, network=network, kind=dto.kind)

        # Two - Phase optimization iterations
        for iteration_id in range(optimization_iterations_count):
            optimizer = BayesianOptimization(
                f=f_,
                pbounds=pbounds,
            )
            optimizer.maximize(
                init_points=1000,
                n_iter=1000,
                acq="ei",
                xi=0.0
            )
            best_params = optimizer.max["params"]
            list_of_best_thresholds = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                args_dict=best_params, network=network, cluster_count=cluster_count, kind=dto.kind)

        print("X")
