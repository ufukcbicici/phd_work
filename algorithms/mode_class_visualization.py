import numpy as np

from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from simple_tf.cign.fast_tree import FastTreeNetwork


class ModeVisualizer:
    def __init__(self, network, dataset, label_list, branch_probs,
                 activations, posterior_probs, degree_list, network_name, run_id, iteration):
        self.dataset = dataset
        self.labelList = label_list
        self.branchProbs = branch_probs
        self.activations = activations
        self.posteriorProbs = posterior_probs
        self.network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name="None", node_costs=None)
        leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
            FastTreeNetwork.load_routing_info(network=network, run_id=run_id, iteration=iteration)
        labels_list = list(leaf_true_labels_dict.values())
        assert all([np.array_equal(labels_list[idx], labels_list[idx + 1]) for idx in range(len(labels_list) - 1)])
        sample_count = label_list.shape[0]
        self.multipathCalculator = MultipathCalculatorV2(
            thresholds_list=None, network=network,
            sample_count=sample_count,
            label_list=label_list, branch_probs=branch_probs_dict,
            activations=activations_dict, posterior_probs=posterior_probs_dict)

    def get_sample_distribution_visual(self):
        thresholds_dict = {}

        # self.multipathCalculator.get_sample_distributions_on_leaf_nodes(thresholds_dict=thr)

def main():
    degree_list = [2, 2]
    node_costs = {i: 1 for i in range(7)}
    run_id = 67
    iteration = 120000

    print("X")
