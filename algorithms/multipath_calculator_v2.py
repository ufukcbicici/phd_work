import threading
import numpy as np
from collections import Counter

from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import MultipathResult, BranchingInfo
from auxillary.general_utility_funcs import UtilityFuncs


class MultipathCalculatorV2:
    def __init__(self, thresholds_list, network):
        self.thresholdsList = thresholds_list
        # self.thresholdsDict structure: (node_index, thresholds[] -> len(child_nodes) sized array.
        # Thresholds are sorted according to the child node indices:
        # i.th child's threshold is at: thresholds[sorted(child_nodes).index_of(child_node[i])]
        self.network = network
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        # self.labelList = label_list
        # self.branchProbs = branch_probs
        # self.activations = activations
        # self.posteriorProbs = posterior_probs
        self.routingResultsDict = {}
        self.networkActivationCosts = {}
        self.kvRows = []
        self.baseEvaluationCost = None
        self.get_evaluation_costs()

    def get_evaluation_costs(self):
        list_of_lists = []
        path_costs = []
        for node in self.leafNodes:
            list_of_lists.append([0, 1])
            leaf_ancestors = self.network.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([self.network.nodeCosts[ancestor.index] for ancestor in leaf_ancestors]))
        self.baseEvaluationCost = np.mean(np.array(path_costs))
        all_result_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=list_of_lists)
        for result_tuple in all_result_tuples:
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(self.leafNodes):
                if result_tuple[node_idx] == 0:
                    continue
                leaf_ancestors = self.network.dagObject.ancestors(node=curr_node)
                leaf_ancestors.append(curr_node)
                for ancestor in leaf_ancestors:
                    processed_nodes_set.add(ancestor.index)
            total_cost = sum([self.network.nodeCosts[n_idx] for n_idx in processed_nodes_set])
            self.networkActivationCosts[result_tuple] = total_cost

    def get_routing_info_from_parent(self, curr_node, branching_info_dict, routing_data):
        sample_count = routing_data.labelList.shape[0]
        if curr_node.isRoot:
            reaches_to_this_node_vector = np.ones(shape=(sample_count,), dtype=np.bool_)
            path_probability = np.ones(shape=(sample_count,))
        else:
            parent_node = self.network.dagObject.parents(node=curr_node)[0]
            siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in
                             enumerate(
                                 sorted(self.network.dagObject.children(node=parent_node),
                                        key=lambda c_node: c_node.index))}
            sibling_index = siblings_dict[curr_node.index]
            reaches_to_this_node_vector = branching_info_dict[parent_node.index].routingMatrix[:, sibling_index]
            path_probability = branching_info_dict[parent_node.index].pathProbabilities[:, sibling_index]
        return reaches_to_this_node_vector, path_probability

    def run(self, routing_data):
        results = []
        self.kvRows = []
        for thresholds_dict in self.thresholdsList:
            res_0, res_1 = self.calculate_for_threshold(thresholds_dict=thresholds_dict, routing_data=routing_data)
            results.append(res_0)
            results.append(res_1)
        return results

    def get_inner_node_routing_info(self, curr_node, routing_data, thresholds_dict, branching_info_dict):
        reaches_to_this_node_vector, path_probability = \
            self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                              routing_data=routing_data)
        p_n_given_x = routing_data.branchProbs[curr_node.index]
        thresholds_matrix = np.zeros_like(p_n_given_x)
        child_nodes = self.network.dagObject.children(node=curr_node)
        child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
        for child_index, child_node in enumerate(child_nodes_sorted):
            thresholds_matrix[:, child_index] = thresholds_dict[curr_node.index][child_index]
        routing_matrix = p_n_given_x >= thresholds_matrix
        routing_matrix = np.logical_and(routing_matrix, np.expand_dims(reaches_to_this_node_vector, axis=1))
        path_probabilities = p_n_given_x * np.expand_dims(path_probability, axis=1)
        branching_info_dict[curr_node.index] = BranchingInfo(branching_probs=p_n_given_x, routing_matrix=routing_matrix,
                                                             path_probs=path_probabilities)

    def get_sample_distributions_on_leaf_nodes(self, thresholds_dict, routing_data, mode_threshold=0.8):
        branching_info_dict = {}
        # Calculate path probabilities
        for curr_node in self.innerNodes:
            self.get_inner_node_routing_info(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                             thresholds_dict=thresholds_dict, routing_data=routing_data)
        leaf_reachability_dict = {}
        for curr_node in self.leafNodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                                  routing_data=routing_data)
            leaf_reachability_dict[curr_node.index] = reaches_to_this_node_vector
        return leaf_reachability_dict

    def calculate_for_threshold(self, thresholds_dict, routing_data):
        branching_info_dict = {}
        sample_count = routing_data.labelList.shape[0]

        def get_computation_cost(leaf_activation):
            return self.networkActivationCosts[tuple(leaf_activation.tolist())]

        # Calculate path probabilities
        for curr_node in self.innerNodes:
            self.get_inner_node_routing_info(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                             thresholds_dict=thresholds_dict, routing_data=routing_data)
        # Calculate averaged posteriors
        posterior_matrices_list = []
        routing_decisions_list = []
        path_probability_list = []
        leaf_index_dict = {}
        for curr_node in self.leafNodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                                  routing_data=routing_data)
            posteriors = np.expand_dims(routing_data.posteriorProbs[curr_node.index], axis=2)
            posterior_matrices_list.append(posteriors)
            leaf_index_dict[len(routing_decisions_list)] = curr_node.index
            routing_decisions_list.append(np.expand_dims(reaches_to_this_node_vector, axis=1))
            path_probability_list.append(np.expand_dims(path_probability, axis=2))
        posteriors_matrix = np.concatenate(posterior_matrices_list, axis=2)
        routing_decisions_matrix = np.concatenate(routing_decisions_list, axis=1).astype(np.float32)
        path_probabilities_matrix = np.concatenate(path_probability_list, axis=1)
        # Method 1: Directly average all valid posteriors
        posteriors_matrix_with_routing = \
            posteriors_matrix * np.expand_dims(routing_decisions_matrix, axis=1)
        posteriors_summed = np.sum(posteriors_matrix_with_routing, axis=2)
        leaf_counts_vector = np.sum(routing_decisions_matrix, axis=1, keepdims=True)
        total_computation_cost = np.sum(np.apply_along_axis(get_computation_cost, axis=1, arr=routing_decisions_matrix))
        posteriors_averaged = posteriors_summed / leaf_counts_vector
        # Method 2: Use a weighted average for all valid posteriors, by using their path probabilities
        path_probabilities_matrix_with_routing = path_probabilities_matrix * routing_decisions_matrix
        path_probabilities_with_routing_sum = np.sum(path_probabilities_matrix_with_routing, axis=1, keepdims=True)
        leaf_weights = path_probabilities_matrix_with_routing / path_probabilities_with_routing_sum
        posteriors_matrix_with_weights = posteriors_matrix * np.expand_dims(leaf_weights, axis=1)
        posteriors_weighted_averaged = np.sum(posteriors_matrix_with_weights, axis=2)
        # Measure accuracies
        simple_avg_predicted_labels = np.argmax(posteriors_averaged, axis=1)
        weighted_avg_predicted_labels = np.argmax(posteriors_weighted_averaged, axis=1)
        total_leaves_evaluated = np.asscalar(np.sum(leaf_counts_vector))
        accuracy_simple_avg = np.sum((simple_avg_predicted_labels == routing_data.labelList).astype(np.float32)) \
                              / float(sample_count)
        accuracy_weighted_avg = np.sum((weighted_avg_predicted_labels == routing_data.labelList).astype(np.float32)) \
                                / float(sample_count)
        computation_overload = total_computation_cost / (sample_count * self.baseEvaluationCost)
        # Tuple: Entry 0: Method Entry 1: Thresholds Entry 2: Accuracy Entry 3: Num of leaves evaluated
        # Entry 4: Computation Overload
        res_method_0 = MultipathResult(result_tuple=(0, thresholds_dict, accuracy_simple_avg,
                                                     total_leaves_evaluated,
                                                     computation_overload))
        res_method_1 = MultipathResult(result_tuple=(1, thresholds_dict, accuracy_weighted_avg,
                                                     total_leaves_evaluated,
                                                     computation_overload))
        return res_method_0, res_method_1
