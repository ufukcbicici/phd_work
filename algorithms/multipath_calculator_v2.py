import threading
import numpy as np


class MultipathCalculatorV2(threading.Thread):
    class BranchingInfo:
        def __init__(self, branching_probs, routing_matrix, path_probs):
            self.branchingProbs = branching_probs
            self.routingMatrix = routing_matrix
            self.pathProbabilities = path_probs

    def __init__(self, thread_id, run_id, iteration, thresholds_list,
                 network, sample_count, label_list, branch_probs, posterior_probs):
        threading.Thread.__init__(self)
        self.threadId = thread_id
        self.runId = run_id
        self.iteration = iteration
        self.thresholdsList = thresholds_list
        # self.thresholdsList structure: (node_index, thresholds[] -> len(child_nodes) sized array.
        # Thresholds are sorted according to the child node indices:
        # i.th child's threshold is at: thresholds[sorted(child_nodes).index_of(child_node[i])]
        self.network = network
        self.sampleCount = sample_count
        self.labelList = label_list
        self.branchProbs = branch_probs
        self.posteriorProbs = posterior_probs
        self.kvRows = []

    def run(self):
        root_node = self.network.nodes[0]
        leaf_count = len([node for node in self.network.topologicalSortedNodes if node.isLeaf])
        max_num_of_samples = leaf_count * self.sampleCount
        root_node = self.network.topologicalSortedNodes[0]
        branching_info_dict = {}
        for curr_node in self.network.topologicalSortedNodes:
            if not curr_node.isLeaf:
                if curr_node.isRoot:
                    reaches_to_this_node_vector = np.ones(shape=(self.sampleCount,), dtype=np.bool_)
                    path_probability = np.ones(shape=(self.sampleCount, ))
                else:
                    parent_node = self.network.parents(node=curr_node)[0]
                    siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in
                                     enumerate(
                                         sorted(self.network.dagObject.children(node=parent_node),
                                                key=lambda c_node: c_node.index))}
                    sibling_index = siblings_dict[curr_node.index]
                    reaches_to_this_node_vector = branching_info_dict[parent_node.index].routingMatrix[:, sibling_index]
                p_n_given_x = self.branchProbs[curr_node.index]
                thresholds_matrix = np.zeros_like(p_n_given_x)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                for child_index, child_node in enumerate(child_nodes_sorted):
                    thresholds_matrix[:, child_index] = self.thresholdsList[curr_node.index][child_index]
                routing_matrix = p_n_given_x >= thresholds_matrix
                routing_matrix = np.logical_and(routing_matrix, np.expand_dims(reaches_to_this_node_vector, axis=1))
                branching_info_dict[curr_node.index] = \
                    MultipathCalculatorV2.BranchingInfo(branching_probs=p_n_given_x, routing_matrix=routing_matrix)

