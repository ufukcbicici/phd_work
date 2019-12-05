import numpy as np

from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import MultipathResult, BranchingInfo
from auxillary.general_utility_funcs import UtilityFuncs


class MultipathCalculatorEarlyExit(MultipathCalculatorV2):
    def __init__(self, network):
        super().__init__(network)
        self.posteriorsEarly = None
        self.posteriorsLate = None

    def get_evaluation_costs(self):
        list_of_lists = []
        path_costs = []
        for node in self.leafNodes:
            path_cost = 0
            list_of_lists.append([0, 1])
            leaf_ancestors = self.network.dagObject.ancestors(node=node)
            path_cost += sum([self.network.nodeCosts[ancestor.index] for ancestor in leaf_ancestors])
            assert all(["early_exit" in k or "late_exit" in k for k in node.opMacCostsDict.keys()])
            leaf_early_exit_cost = sum([node.opMacCostsDict[k] for k in node.opMacCostsDict.keys()
                                        if "early_exit" in k])
            path_cost += leaf_early_exit_cost
            path_costs.append(path_cost)
        self.baseEvaluationCost = np.mean(np.array(path_costs))
        early_exit_costs_dict = {}
        late_exit_costs_dict = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                leaf_early_exit_cost = sum([node.opMacCostsDict[k] for k in node.opMacCostsDict.keys()
                                            if "early_exit" in k])
                leaf_late_exit_cost = sum([node.opMacCostsDict[k] for k in node.opMacCostsDict.keys()
                                           if "late_exit" in k])
                early_exit_costs_dict[node.index] = leaf_early_exit_cost
                late_exit_costs_dict[node.index] = leaf_late_exit_cost
            else:
                early_exit_costs_dict[node.index] = self.network.nodeCosts[node.index]
                late_exit_costs_dict[node.index] = self.network.nodeCosts[node.index]
        all_result_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=list_of_lists)
        for result_tuple in all_result_tuples:
            cost_dict = early_exit_costs_dict if sum(result_tuple) == 1 else late_exit_costs_dict
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(self.leafNodes):
                if result_tuple[node_idx] == 0:
                    continue
                leaf_ancestors = self.network.dagObject.ancestors(node=curr_node)
                leaf_ancestors.append(curr_node)
                for ancestor in leaf_ancestors:
                    processed_nodes_set.add(ancestor.index)
            total_cost = sum([cost_dict[n_idx] for n_idx in processed_nodes_set])
            self.networkActivationCosts[result_tuple] = total_cost

    def calculate_for_threshold(self, thresholds_dict, routing_data):
        self.branchProbs = routing_data.get_dict("branch_probs")
        self.posteriorsEarly = routing_data.get_dict("posterior_probs")
        self.posteriorsLate = routing_data.get_dict("posterior_probs_late")
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
        early_posterior_matrices_list = []
        late_posterior_matrices_list = []
        routing_decisions_list = []
        path_probability_list = []
        for curr_node in self.leafNodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict)
            posteriors_early = np.expand_dims(self.posteriorsEarly[curr_node.index], axis=2)
            early_posterior_matrices_list.append(posteriors_early)
            posteriors_late = np.expand_dims(self.posteriorsLate[curr_node.index], axis=2)
            late_posterior_matrices_list.append(posteriors_late)
            assert len(reaches_to_this_node_vector.shape) == 1
            assert len(path_probability)
            routing_decisions_list.append(np.expand_dims(reaches_to_this_node_vector, axis=1))
            path_probability_list.append(np.expand_dims(path_probability, axis=1))
        routing_decisions_matrix = np.concatenate(routing_decisions_list, axis=1).astype(np.float32)
        path_probabilities_matrix = np.concatenate(path_probability_list, axis=1)
        posteriors_matrix_early = np.concatenate(early_posterior_matrices_list, axis=2)
        posteriors_matrix_late = np.concatenate(late_posterior_matrices_list, axis=2)
        indicator_vec = np.sum(routing_decisions_matrix, axis=1) == 1.0
        posteriors_matrix = np.where(np.reshape(indicator_vec, (indicator_vec.shape[0], 1, 1)),
                                     posteriors_matrix_early, posteriors_matrix_late)
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
