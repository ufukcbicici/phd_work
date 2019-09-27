from simple_tf.cign.fast_tree import FastTreeNetwork


class ModeVisualizer:
    def __init__(self, network, dataset, label_list, branch_probs,
                 activations, posterior_probs):
        self.network = network
        self.dataset = dataset
        self.labelList = label_list
        self.branchProbs = branch_probs
        self.activations = activations
        self.posteriorProbs = posterior_probs


def main():
    node_costs = {i: 1 for i in range(7)}
    run_id = 67
    iteration = 120000

    tree = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name="None", node_costs=node_costs)
    leaf_true_labels_dict, branch_probs_dict, posterior_probs_dict, activations_dict = \
        FastTreeNetwork.load_routing_info(network=tree, run_id=run_id, iteration=iteration)

    print("X")

