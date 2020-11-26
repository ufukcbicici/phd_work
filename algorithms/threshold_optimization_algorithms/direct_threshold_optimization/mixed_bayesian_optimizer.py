import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

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
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        list_of_threshold_dicts = []
        for cluster_id in range(cluster_count):
            thresholds_dict = {}
            for node in inner_nodes:
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
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        pbounds = {}
        for cluster_id in range(cluster_count):
            for node in inner_nodes:
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
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        list_of_threshold_dicts = []
        for cluster_id in range(cluster_count):
            thrs_dict = {}
            for node in inner_nodes:
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
    def optimize(cluster_count, fc_layers, run_id, network, iteration, routing_data, seed, test_ratio):
        indices = np.arange(routing_data.dictOfDatasets[iteration].labelList.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=test_ratio)
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        leaf_nodes = [node for node in network.topologicalSortedNodes if node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        leaf_nodes = sorted(leaf_nodes, key=lambda node: node.index)
        leaf_indices = {node.index: idx for idx, node in enumerate(leaf_nodes)}
        label_count = len(set(routing_data.dictOfDatasets[routing_data.iterations[0]].labelList))
        # dto = DirectThresholdOptimizer(network=network, routing_data=routing_data, iteration=iteration, seed=seed,
        #                                train_indices=train_indices, test_indices=test_indices)
        # Threshold Optimizer
        dto = DirectThresholdOptimizerEntropy(network=network, routing_data=routing_data, iteration=iteration,
                                              seed=seed,
                                              train_indices=train_indices, test_indices=test_indices)
        temperatures_dict = dto.calibrate_branching_probabilities(run_id=run_id, iteration=iteration, seed=seed)
        dto.build_network()
        # Clusterer
        bc = BayesianClusterer(network=network, routing_data=routing_data,
                               cluster_count=cluster_count, fc_layers=fc_layers)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        mixing_lambda = 1.0

        # Loss Function
        def f_(**kwargs):
            list_of_threshold_dicts = MixedBayesianOptimizer.decode_bayesian_optimization_parameters(
                args_dict=kwargs, network=network, cluster_count=cluster_count, kind=dto.kind)





            # Reconstruct the thresholds dict
            # thrs_dict = {}
            # for node in inner_nodes:
            #     child_nodes = network.dagObject.children(node)
            #     child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
            #     if dto.kind == "probabiliy":
            #         thrs_arr = np.array([kwargs["t_{0}_{1}".format(node.index, c_nd.index)] for c_nd in child_nodes])
            #         thrs_dict[node.index] = thrs_arr[np.newaxis, :]
            #     elif dto.kind == "entropy":
            #         thrs_dict[node.index] = kwargs["t_{0}".format(node.index)]
            #     else:
            #         raise NotImplementedError()
            # # Calculate the score
            # scr = dto.measure_score(sess=sess, indices=train_indices,
            #                         iteration=iteration, temperatures_dict=temperatures_dict,
            #                         thresholds_dict=thrs_dict, mixing_lambda=mixing_lambda)
            # return scr

        pbounds = MixedBayesianOptimizer.calculate_bounds(network=network, kind=dto.kind)
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
        print("X")
