import numpy as np
import tensorflow as tf


class MultiIterationDQN:
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names):
        self.routingDataset = routing_dataset
        self.network = network
        self.networkName = network_name
        self.runId = run_id
        self.usedFeatureNames = used_feature_names
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        # Data dictionaries
        self.maxLikelihoodPaths = {}
        # Init data structures
        self.get_max_likelihood_paths_of_iterations()
        # The following is for testing, can comment out later.
        self.test_likelihood_consistency()
        print("X")

    def get_max_likelihood_paths_of_iterations(self):
        for iteration in self.routingDataset.iterations:
            self.maxLikelihoodPaths[iteration] = self.get_max_likelihood_paths(
                branch_probs=self.routingDataset.dictOfDatasets[iteration].get_dict("branch_probs"))

    def get_max_likelihood_paths(self, branch_probs):
        sample_sizes = list(set([arr.shape[0] for arr in branch_probs.values()]))
        assert len(sample_sizes) == 1
        sample_size = sample_sizes[0]
        max_likelihood_paths = []
        for idx in range(sample_size):
            curr_node = self.network.topologicalSortedNodes[0]
            route = []
            while True:
                route.append(curr_node.index)
                if curr_node.isLeaf:
                    break
                routing_distribution = branch_probs[curr_node.index][idx]
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
            max_likelihood_paths.append(np.array(route))
        max_likelihood_paths = np.stack(max_likelihood_paths, axis=0)
        return max_likelihood_paths

    # Test methods
    def test_likelihood_consistency(self):
        for idx in range(self.routingDataset.labelList.shape[0]):
            path_array = []
            for iteration in self.routingDataset.iterations:
                iteration_id = self.routingDataset.linkageInfo[(idx, iteration)]
                path_array.append(self.maxLikelihoodPaths[iteration][iteration_id])
            path_array = np.stack(path_array, axis=0)
            print("X")

