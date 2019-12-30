import numpy as np

from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import MultipathResult, BranchingInfo
from auxillary.general_utility_funcs import UtilityFuncs


class MultipathCalculatorV2:
    def __init__(self, network, routing_data):
        # self.thresholdsDict structure: (node_index, thresholds[] -> len(child_nodes) sized array.
        # Thresholds are sorted according to the child node indices:
        # i.th child's threshold is at: thresholds[sorted(child_nodes).index_of(child_node[i])]
        self.network = network
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        # self.labelList = label_list
        # self.branchProbs = branch_probs
        # self.activations = activations
        # self.posteriorProbs = posterior_probs
        self.routingResultsDict = {}
        self.networkActivationCosts = {}
        self.kvRows = []
        self.baseEvaluationCost = None
        self.branchProbs = routing_data.get_dict("branch_probs")
        self.posteriors = routing_data.get_dict("posterior_probs")
        self.labelList = routing_data.labelList
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

    def get_routing_info_from_parent(self, curr_node, branching_info_dict):
        sample_count = self.labelList.shape[0]
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

    def get_inner_node_routing_info(self, curr_node, thresholds_dict, branching_info_dict):
        reaches_to_this_node_vector, path_probability = \
            self.get_routing_info_from_parent(curr_node=curr_node,
                                              branching_info_dict=branching_info_dict)
        p_n_given_x = self.branchProbs[curr_node.index]
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

    def get_sample_distributions_on_leaf_nodes(self, thresholds_dict, mode_threshold=0.8):
        branching_info_dict = {}
        # Calculate path probabilities
        for curr_node in self.innerNodes:
            self.get_inner_node_routing_info(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                             thresholds_dict=thresholds_dict)
        leaf_reachability_dict = {}
        for curr_node in self.leafNodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict)
            leaf_reachability_dict[curr_node.index] = reaches_to_this_node_vector
        return leaf_reachability_dict

    def calculate_for_threshold(self, thresholds_dict, routing_data):
        self.branchProbs = routing_data.get_dict("branch_probs")
        self.posteriors = routing_data.get_dict("posterior_probs")
        self.labelList = routing_data.labelList

        branching_info_dict = {}
        sample_count = self.labelList.shape[0]

        def get_computation_cost(leaf_activation):
            return self.networkActivationCosts[tuple(leaf_activation.tolist())]

        # Calculate path probabilities
        for curr_node in self.innerNodes:
            self.get_inner_node_routing_info(curr_node=curr_node, branching_info_dict=branching_info_dict,
                                             thresholds_dict=thresholds_dict)
        # Calculate averaged posteriors
        posterior_matrices_list = []
        routing_decisions_list = []
        path_probability_list = []
        for curr_node in self.leafNodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict)
            posteriors = np.expand_dims(self.posteriors[curr_node.index], axis=2)
            posterior_matrices_list.append(posteriors)
            assert len(reaches_to_this_node_vector.shape) == 1
            assert len(path_probability)
            routing_decisions_list.append(np.expand_dims(reaches_to_this_node_vector, axis=1))
            path_probability_list.append(np.expand_dims(path_probability, axis=1))
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
        accuracy_simple_avg = np.sum((simple_avg_predicted_labels == self.labelList).astype(np.float32)) \
                              / float(sample_count)
        accuracy_weighted_avg = np.sum((weighted_avg_predicted_labels == self.labelList).astype(np.float32)) \
                                / float(sample_count)
        computation_overload = total_computation_cost / (sample_count * self.baseEvaluationCost)
        # Tuple: Entry 0: Method Entry 1: Thresholds Entry 2: Accuracy Entry 3: Num of leaves evaluated
        # Entry 4: Computation Overload
        res_method_0 = MultipathResult(result_tuple=(0, thresholds_dict, accuracy_simple_avg,
                                                     total_leaves_evaluated,
                                                     computation_overload,
                                                     routing_decisions_matrix, posteriors_averaged))
        res_method_1 = MultipathResult(result_tuple=(1, thresholds_dict, accuracy_weighted_avg,
                                                     total_leaves_evaluated,
                                                     computation_overload,
                                                     routing_decisions_matrix, posteriors_weighted_averaged))

        self.branchProbs = None
        self.posteriors = None
        self.labelList = None
        return res_method_0, res_method_1
