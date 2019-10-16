import numpy as np

from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cign.fast_tree import FastTreeNetwork


class ModeVisualizer:
    def __init__(self, network, dataset, run_id, iteration):
        self.dataset = dataset
        leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
            FastTreeNetwork.load_routing_info(network=network, run_id=run_id, iteration=iteration)
        labels_list = list(leaf_true_labels_dict.values())
        assert all([np.array_equal(labels_list[idx], labels_list[idx + 1]) for idx in range(len(labels_list) - 1)])
        label_list = labels_list[0]
        sample_count = label_list.shape[0]
        self.multipathCalculator = MultipathCalculatorV2(
            thresholds_list=None, network=network,
            sample_count=sample_count,
            label_list=label_list, branch_probs=branch_probs_dict,
            activations=activations_dict, posterior_probs=posterior_probs_dict)

    def get_sample_distribution_visual(self, network):
        threshold_state = {}
        for node in network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            threshold_state[node.index] = max_threshold * np.ones(shape=(child_count, ))
        self.multipathCalculator.get_sample_distributions_on_leaf_nodes(thresholds_dict=threshold_state)

def main():
    run_id = 67
    # network_name = "Cifar100_CIGN_Sampling"
    network_name = "None"
    iterations = [119100]
    node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
    dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
    mode_visualizer = ModeVisualizer(network=tree, dataset=dataset, run_id=67, iteration=119100)
    mode_visualizer.get_sample_distribution_visual(network=tree)
    print("X")
