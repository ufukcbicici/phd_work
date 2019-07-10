from algorithms.simple_accuracy_calculator import SimpleAccuracyCalculator
from simple_tf.cign.fast_tree import FastTreeNetwork

run_id = 789
iterations = [43680 + i*480 for i in range(10)]


def main():
    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2])
    labels_collection = {}
    branch_probs_collection = {}
    posteriors_collection = {}
    activations_collection = {}
    for iteration in iterations:
        leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
            SimpleAccuracyCalculator.load_routing_info(network=tree, run_id=run_id, iteration=iteration)
        labels_collection[iteration] = leaf_true_labels_dict
        branch_probs_collection[iteration] = branch_probs_dict
        posteriors_collection[iteration] = posterior_probs_dict
        activations_collection[iteration] = activations_dict
    print("X")


