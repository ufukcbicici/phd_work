import numpy as np


class InformationGainRoutingAccuracyCalculator:
    def __init__(self):
        pass

    @staticmethod
    def get_max_likelihood_paths(network, routing_data):
        branch_probs = routing_data.get_dict("branch_probs")
        sample_sizes = list(set([arr.shape[0] for arr in branch_probs.values()]))
        assert len(sample_sizes) == 1
        sample_size = sample_sizes[0]
        max_likelihood_paths = []
        for idx in range(sample_size):
            curr_node = network.topologicalSortedNodes[0]
            route = []
            while True:
                route.append(curr_node.index)
                if curr_node.isLeaf:
                    break
                routing_distribution = branch_probs[curr_node.index][idx]
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
            max_likelihood_paths.append(np.array(route))
        max_likelihood_paths = np.stack(max_likelihood_paths, axis=0)
        return max_likelihood_paths

    @staticmethod
    def get_ig_posteriors(network, routing_data, indices):
        min_leaf_id = min([node.index for node in network.orderedNodesPerLevel[network.depth - 1]])
        posteriors = \
            np.stack([routing_data.get_dict("posterior_probs")[node.index] for node in network.leafNodes], axis=2)
        posteriors = posteriors[indices]
        max_likelihood_paths = InformationGainRoutingAccuracyCalculator. \
            get_max_likelihood_paths(network=network,
                                     routing_data=routing_data)
        ml_indices = max_likelihood_paths[indices][:, -1] - min_leaf_id
        selected_posteriors = posteriors[np.arange(posteriors.shape[0]), :, ml_indices]
        return selected_posteriors

    @staticmethod
    def calculate(network, routing_data, indices):
        selected_posteriors = InformationGainRoutingAccuracyCalculator.\
            get_ig_posteriors(network=network, routing_data=routing_data, indices=indices)
        true_labels = routing_data.labelList[indices]
        predicted_labels = np.argmax(selected_posteriors, axis=1)
        correct_count = np.sum((predicted_labels == true_labels).astype(np.float32))
        accuracy = correct_count / true_labels.shape[0]
        return accuracy

    @staticmethod
    def calculate_for_ensembles(list_of_networks, list_of_routing_data, indices):
        posteriors_tensor = []
        true_labels = list_of_routing_data[0].labelList[indices]
        for network, routing_data in zip(list_of_networks, list_of_routing_data):
            selected_posteriors = InformationGainRoutingAccuracyCalculator. \
                get_ig_posteriors(network=network, routing_data=routing_data, indices=indices)
            posteriors_tensor.append(selected_posteriors)
        posteriors_tensor = np.stack(posteriors_tensor, axis=-1)
        ensemble_posterior = np.mean(posteriors_tensor, axis=-1)
        predicted_labels = np.argmax(ensemble_posterior, axis=1)
        correct_count = np.sum((predicted_labels == true_labels).astype(np.float32))
        accuracy = correct_count / true_labels.shape[0]
        return accuracy
