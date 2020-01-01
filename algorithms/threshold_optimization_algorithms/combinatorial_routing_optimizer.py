import numpy as np
import pickle
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class CombinatorialRoutingOptimizer:
    def __init__(self, network_name, run_id, iteration, degree_list, data_type, output_names):
        self.network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        self.routingData = self.network.load_routing_info(run_id=run_id, iteration=iteration, data_type=data_type,
                                                          output_names=output_names)
        self.branchProbs = self.routingData.get_dict("branch_probs")
        self.posteriors = self.routingData.get_dict("posterior_probs")
        self.labelsVector = self.routingData.labelList
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]

    def calculate_best_routes(self):
        total_correct_count = 0
        single_path_correct_count = 0
        posteriors_tensor = np.stack([self.posteriors[node.index] for node in self.leafNodes], axis=2)
        for idx in range(self.labelsVector.shape[0]):
            posteriors_matrix = posteriors_tensor[idx]
            curr_node = self.network.topologicalSortedNodes[0]
            route = []
            max_likelihood_leaf = None
            while True:
                route.append(curr_node.index)
                if curr_node.isLeaf:
                    max_likelihood_leaf = curr_node
                    break
                routing_distribution = self.branchProbs[curr_node.index][idx]
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
            single_path_correct_count += float(np.argmax(
                posteriors_matrix[:, self.leafIndices[max_likelihood_leaf.index]]) == self.labelsVector[idx])
            valid_routes = set()
            for r in self.routingCombinations:
                new_route = r.copy()
                new_route[self.leafIndices[max_likelihood_leaf.index]] = 1
                valid_routes.add(tuple(new_route.tolist()))
            is_correct = False
            for valid_route in valid_routes:
                uniform_weight = 1.0 / sum(valid_route)
                posteriors_sparse = posteriors_matrix * (uniform_weight * np.expand_dims(np.array(valid_route), axis=0))
                posteriors_weighted = np.sum(posteriors_sparse, axis=1)
                if np.argmax(posteriors_weighted) == self.labelsVector[idx]:
                    is_correct = True
            total_correct_count += float(is_correct)
        print("X")


def main():
    run_id = 67
    # network_name = "Cifar100_CIGN_Sampling"
    network_name = "None"
    iteration = 119100
    # node_costs = {0: 67391424.0, 2: 16754176.0, 6: 3735040.0, 5: 3735040.0, 1: 16754176.0, 4: 3735040.0, 3: 3735040.0}
    # pickle.dump(node_costs, open("nodeCosts.sav", "wb"))
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs"]
    # dataset = CifarDataSet(session=None, validation_sample_count=0, load_validation_from=None)
    # whitened_images = RoutingVisualizer.whiten_dataset(dataset=dataset)
    routing_optimizer = CombinatorialRoutingOptimizer(network_name=network_name, run_id=run_id, iteration=iteration,
                                                      degree_list=[2, 2], data_type="", output_names=output_names)
    routing_optimizer.calculate_best_routes()


main()