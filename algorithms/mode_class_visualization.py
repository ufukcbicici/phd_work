import numpy as np
import matplotlib.pyplot as plt
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cign.fast_tree import FastTreeNetwork
from collections import Counter


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

    def get_sample_distribution_visual(self, network, dataset, mode_threshold=0.8, sample_count_per_class=5):
        threshold_state = {}
        for node in network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            threshold_state[node.index] = max_threshold * np.ones(shape=(child_count, ))
        leaf_reachability_dict = \
            self.multipathCalculator.get_sample_distributions_on_leaf_nodes(thresholds_dict=threshold_state)
        label_list = self.multipathCalculator.labelList
        num_of_labels = len(set(label_list.tolist()))
        # Calculate mode distributions
        for node in network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            reached_labels = label_list[leaf_reachability_dict[node.index]]
            counter = Counter(reached_labels)
            label_freq_pairs = [(label, float(count) / float(reached_labels.shape[0]))
                                for label, count in counter.items()]
            label_freq_pairs = sorted(label_freq_pairs, key=lambda tpl: tpl[1], reverse=True)
            cut_off_idx = 0
            cumulative_probability = 0
            while True:
                new_cumul_prob = cumulative_probability + label_freq_pairs[cut_off_idx][1]
                if new_cumul_prob >= mode_threshold:
                    break
                cut_off_idx += 1
                cumulative_probability = new_cumul_prob
            mode_labels = label_freq_pairs[0: cut_off_idx]
            self.plot_mode_images(dataset=dataset, node=node, mode_labels=mode_labels,
                                  sample_count_per_class=sample_count_per_class)
            print("X")

    def plot_mode_images(self, dataset, node, mode_labels, sample_count_per_class):
        # Design the plot
        fig, axs = plt.subplots(len(mode_labels), sample_count_per_class)
        fig.suptitle('Mode Labels for Leaf{0}'.format(node.index))
        for row, tpl in enumerate(mode_labels):
            label = tpl[0]
            probability_mass = tpl[1]
            sample_indices_with_correct_label = np.nonzero(dataset.testLabels == label)[0]
            # Pick random indices
            indices = np.random.choice(sample_indices_with_correct_label, sample_count_per_class, replace=False)
            for col in range(sample_count_per_class):
                img = dataset.testSamples[indices[col]]
                axs[row, col].imshow(img)
                axs[row, col].set_title("Label:{0} Prob Mass:{1}".format(label, probability_mass))
        plt.show()


def main():
    run_id = 67
    # network_name = "Cifar100_CIGN_Sampling"
    network_name = "None"
    iterations = [119100]
    node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name, node_costs=node_costs)
    dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
    mode_visualizer = ModeVisualizer(network=tree, dataset=dataset, run_id=67, iteration=119100)
    mode_visualizer.get_sample_distribution_visual(network=tree, dataset=dataset)
    print("X")
