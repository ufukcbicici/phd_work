import threading
import numpy as np


class MultipathCalculatorV2(threading.Thread):
    class BranchingInfo:
        def __init__(self, branching_probs, routing_matrix, path_probs):
            self.branchingProbs = branching_probs
            self.routingMatrix = routing_matrix
            self.pathProbabilities = path_probs

    def __init__(self, thread_id, run_id, iteration, thresholds_list,
                 network, sample_count, label_list, branch_probs, activations, posterior_probs):
        threading.Thread.__init__(self)
        self.threadId = thread_id
        self.runId = run_id
        self.iteration = iteration
        self.thresholdsList = thresholds_list
        # self.thresholdsDict structure: (node_index, thresholds[] -> len(child_nodes) sized array.
        # Thresholds are sorted according to the child node indices:
        # i.th child's threshold is at: thresholds[sorted(child_nodes).index_of(child_node[i])]
        self.network = network
        self.sampleCount = sample_count
        self.labelList = label_list
        self.branchProbs = branch_probs
        self.activations = activations
        self.posteriorProbs = posterior_probs
        self.kvRows = []

    def get_routing_info_from_parent(self, curr_node, branching_info_dict):
        if curr_node.isRoot:
            reaches_to_this_node_vector = np.ones(shape=(self.sampleCount,), dtype=np.bool_)
            path_probability = np.ones(shape=(self.sampleCount,))
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

    def run(self):
        for thresholds_dict in self.thresholdsList:
            self.calculate_for_threshold(thresholds_dict=thresholds_dict)

    def calculate_for_threshold(self, thresholds_dict):
        root_node = self.network.nodes[0]
        leaf_count = len([node for node in self.network.topologicalSortedNodes if node.isLeaf])
        max_num_of_samples = leaf_count * self.sampleCount
        root_node = self.network.topologicalSortedNodes[0]
        branching_info_dict = {}
        inner_nodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        leaf_nodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        # Calculate path probabilities
        for curr_node in inner_nodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict)
            p_n_given_x = self.branchProbs[curr_node.index]
            thresholds_matrix = np.zeros_like(p_n_given_x)
            child_nodes = self.network.dagObject.children(node=curr_node)
            child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
            for child_index, child_node in enumerate(child_nodes_sorted):
                thresholds_matrix[:, child_index] = thresholds_dict[curr_node.index][child_index]
            routing_matrix = p_n_given_x >= thresholds_matrix
            routing_matrix = np.logical_and(routing_matrix, np.expand_dims(reaches_to_this_node_vector, axis=1))
            path_probabilities = p_n_given_x * np.expand_dims(path_probability, axis=1)
            branching_info_dict[curr_node.index] = \
                MultipathCalculatorV2.BranchingInfo(branching_probs=p_n_given_x, routing_matrix=routing_matrix,
                                                    path_probs=path_probabilities)
        # Calculate averaged posteriors
        posterior_matrices_list = []
        routing_decisions_list = []
        path_probability_list = []
        for curr_node in leaf_nodes:
            reaches_to_this_node_vector, path_probability = \
                self.get_routing_info_from_parent(curr_node=curr_node, branching_info_dict=branching_info_dict)
            posteriors = np.expand_dims(self.posteriorProbs[curr_node.index], axis=2)
            posterior_matrices_list.append(posteriors)
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
                              / float(self.sampleCount)
        accuracy_weighted_avg = np.sum((weighted_avg_predicted_labels == self.labelList).astype(np.float32)) \
                              / float(self.sampleCount)
        print(
            "******* Multipath Threshold:{0} Simple Accuracy:{1} "
            "Weighted Accuracy:{2} Total Leaves Evaluated:{3}*******"
                .format(thresholds_dict, accuracy_simple_avg, accuracy_weighted_avg, total_leaves_evaluated))
        # Temporary
        path_threshold = thresholds_dict[0][0]
        self.kvRows.append((self.runId, self.iteration, 0, path_threshold, accuracy_simple_avg,
                            total_leaves_evaluated))
        self.kvRows.append((self.runId, self.iteration, 1, path_threshold, accuracy_weighted_avg,
                            total_leaves_evaluated))