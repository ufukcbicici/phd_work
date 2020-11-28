import numpy as np


class CignActivationCostCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_mac_cost(network, node_costs):
        path_costs = []
        for node in network.leafNodes:
            leaf_ancestors = network.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([node_costs[ancestor.index] for ancestor in leaf_ancestors]))
        base_evaluation_cost = np.mean(np.array(path_costs))
        network_activation_costs = []
        network_activation_costs_dict = {}
        action_count = 2 ** len(network.leafNodes) - 1
        for action_id in range(action_count):
            l = [int(x) for x in list('{0:0b}'.format(action_id + 1))]
            k = [0] * (len(network.leafNodes) - len(l))
            k.extend(l)
            node_selection = np.array(k)
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(network.leafNodes):
                if node_selection[node_idx] == 0:
                    continue
                leaf_ancestors = network.dagObject.ancestors(node=curr_node)
                leaf_ancestors.append(curr_node)
                for ancestor in leaf_ancestors:
                    processed_nodes_set.add(ancestor.index)
            total_cost = sum([node_costs[n_idx] for n_idx in processed_nodes_set])
            network_activation_costs.append(total_cost)
            network_activation_costs_dict[tuple(node_selection)] = (total_cost / base_evaluation_cost) - 1.0
        network_activation_costs = (np.array(network_activation_costs) * (1.0 / base_evaluation_cost)) - 1.0
        return network_activation_costs, network_activation_costs_dict

    @staticmethod
    def calculate_average_parameter_count(network, node_costs, selections_list, selection_tuples):
        selection_costs_dict = {}
        for selection_id, selection_tuple in selection_tuples:
            selected_set = set()
            for leaf_id, leaf_node in enumerate(network.leafNodes):
                if leaf_id == 0:
                    continue
                leaf_ancestors = network.dagObject.ancestors(node=leaf_node)
                leaf_ancestors.append(leaf_node)
                for ancestor in leaf_ancestors:
                    selected_set.add(ancestor.index)
            selection_costs_dict[selection_id] = np.sum([node_costs[node_id] for node_id in selected_set])
        cost_list = [selection_costs_dict[s_id] for s_id in selections_list]
        average_cost = np.mean(np.array(cost_list))
        return average_cost
