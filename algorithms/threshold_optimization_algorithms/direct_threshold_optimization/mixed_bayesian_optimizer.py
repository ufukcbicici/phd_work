import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer


class MixedBayesianOptimizer:

    @staticmethod
    def get_random_probability_thresholds(network, inner_nodes):
        thresholds_dict = {}
        for node in inner_nodes:
            child_count = len(network.dagObject.children(node))
            thresholds_dict[node.index] = np.random.uniform(low=0.0, high=1.0 / child_count, size=(1, child_count))
        return thresholds_dict

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
        dto = DirectThresholdOptimizer(network=network, routing_data=routing_data, seed=seed)
        temperatures_dict = dto.calibrate_branching_probabilities(run_id=run_id, iteration=iteration, seed=seed)
        dto.build_network()
        sess = tf.Session()

        thresholds_dict = MixedBayesianOptimizer.get_random_probability_thresholds(
            network=network, inner_nodes=inner_nodes)
        accuracy = dto.measure_score(sess=sess, indices=train_indices,
                                     iteration=iteration, temperatures_dict=temperatures_dict,
                                     thresholds_dict=thresholds_dict)
        print("X")
