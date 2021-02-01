import numpy as np


class CignReachabilityMatricesCalculation:
    def __init__(self):
        pass

    @staticmethod
    def calculate_reachibility_matrices(network, action_spaces):
        reachability_matrices = []
        max_trajectory_length = network.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            if t == 0:
                reachability_matrix_t = np.ones(shape=(1, action_spaces[t].shape[0]), dtype=np.int32)
            else:
                reachability_matrix_t = np.zeros(shape=(action_spaces[t - 1].shape[0], action_spaces[t].shape[0]),
                                                 dtype=np.int32)
                for action_t_minus_one_id in range(action_spaces[t - 1].shape[0]):
                    node_selection_vec_t_minus_one = action_spaces[t - 1][action_t_minus_one_id]
                    selected_nodes_t = [node for i, node in enumerate(network.orderedNodesPerLevel[t])
                                        if node_selection_vec_t_minus_one[i] != 0]
                    next_level_nodes = network.orderedNodesPerLevel[t + 1]
                    reachable_next_level_node_ids = set()
                    next_level_reached_dict = {}
                    for parent_node in selected_nodes_t:
                        child_nodes = {c_node.index for c_node in network.dagObject.children(node=parent_node)}
                        reachable_next_level_node_ids = reachable_next_level_node_ids.union(child_nodes)
                        next_level_reached_dict[parent_node.index] = child_nodes

                    for actions_t_id in range(action_spaces[t].shape[0]):
                        # All selected nodes should have their parents selected in the previous depth
                        node_selection_vec_t = action_spaces[t][actions_t_id]
                        reached_nodes = {node.index for is_reached, node in zip(node_selection_vec_t, next_level_nodes)
                                         if is_reached != 0}
                        is_valid_selection = int(len(reached_nodes.difference(reachable_next_level_node_ids)) == 0)
                        # All selected nodes in the previous depth must have at least one child selected in next depth
                        for parent_node in selected_nodes_t:
                            selection_arr = [_n in reached_nodes for _n in next_level_reached_dict[parent_node.index]]
                            is_valid_selection = is_valid_selection and any(selection_arr)
                        reachability_matrix_t[action_t_minus_one_id, actions_t_id] = is_valid_selection
            reachability_matrices.append(reachability_matrix_t)
        return reachability_matrices
