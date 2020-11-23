import numpy as np


class CignActivationCostCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate(network, node_costs):
        inner_nodes = [node for node in network.topologicalSortedNodes if not node.isLeaf]
        leaf_nodes = [node for node in network.topologicalSortedNodes if node.isLeaf]
        inner_nodes = sorted(inner_nodes, key=lambda node: node.index)
        leaf_nodes = sorted(leaf_nodes, key=lambda node: node.index)

        path_costs = []
        for node in leaf_nodes:
            leaf_ancestors = network.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([node_costs[ancestor.index] for ancestor in leaf_ancestors]))
        base_evaluation_cost = np.mean(np.array(path_costs))
        network_activation_costs = []
        network_activation_costs_dict = {}
        action_count = 2 ** len(leaf_nodes) - 1
        for action_id in range(action_count):
            l = [int(x) for x in list('{0:0b}'.format(action_id + 1))]
            k = [0] * (len(leaf_nodes) - len(l))
            k.extend(l)
            node_selection = np.array(k)
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(leaf_nodes):
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
