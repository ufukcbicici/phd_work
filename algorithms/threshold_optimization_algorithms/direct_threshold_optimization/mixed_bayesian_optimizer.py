import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy


class MixedBayesianOptimizer:

    @staticmethod
    def get_random_thresholds(network, kind):
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        thresholds_dict = {}
        for node in inner_nodes:
            child_count = len(network.dagObject.children(node))
            if kind == "probability":
                max_bound = 1.0 / child_count
                thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound, size=(1, child_count))
            elif kind == "entropy":
                max_bound = -np.log(1.0 / child_count)
                thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound)
        return thresholds_dict

    @staticmethod
    def calculate_bounds(network, kind):
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        pbounds = {}
        for node in inner_nodes:
            child_nodes = network.dagObject.children(node)
            child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
            if kind == "probability":
                max_bound = 1.0 / len(child_nodes)
            elif kind == "entropy":
                max_bound = -np.log(1.0 / len(child_nodes))
            else:
                raise NotImplementedError()
            for c_nd in child_nodes:
                pbounds["t_{0}_{1}".format(node.index, c_nd.index)] = (0.0, max_bound)
        return pbounds

    @staticmethod
    def optimize(run_id, network, iteration, routing_data, seed, test_ratio):
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
        dto = DirectThresholdOptimizerEntropy(network=network, routing_data=routing_data, iteration=iteration,
                                              seed=seed,
                                              train_indices=train_indices, test_indices=test_indices)
        temperatures_dict = dto.calibrate_branching_probabilities(run_id=run_id, iteration=iteration, seed=seed)
        dto.build_network()
        sess = tf.Session()

        mixing_lambda = 1.0
        scores = []
        for i in range(100):
            thresholds_dict = MixedBayesianOptimizer.get_random_thresholds(network=network, kind="entropy")
            score = dto.measure_score(sess=sess, indices=train_indices,
                                      iteration=iteration,
                                      mixing_lambda=mixing_lambda,
                                      temperatures_dict=temperatures_dict,
                                      thresholds_dict=thresholds_dict)
            scores.append(score)

        def f_(**kwargs):
            # Reconstruct the thresholds dict
            thrs_dict = {}
            for node in inner_nodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                thrs_arr = np.array([kwargs["t_{0}_{1}".format(node.index, c_nd.index)] for c_nd in child_nodes])
                thrs_dict[node.index] = thrs_arr[np.newaxis, :]
            # Calculate the score
            scr = dto.measure_score(sess=sess, indices=train_indices,
                                    iteration=iteration, temperatures_dict=temperatures_dict,
                                    thresholds_dict=thrs_dict, mixing_lambda=mixing_lambda)
            return scr

        pbounds = MixedBayesianOptimizer.calculate_bounds(network=network, kind="entropy")
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
