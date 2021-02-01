import numpy as np
import tensorflow as tf

from algorithms.cign_activation_cost_calculator import CignActivationCostCalculator
from algorithms.cign_reachbility_matrices_calculation import CignReachabilityMatricesCalculation
from simple_tf.cign.fast_tree import FastTreeNetwork


class CignWithRlRouting(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.actionSpaces = []
        self.networkActivationCosts = None
        self.networkActivationCostsDict = None
        self.reachabilityMatrices = []

    def build_network(self):
        # Regular CIGN stuff here
        super().build_network()
        # Reinforcement Learning things here
        self.build_action_spaces()
        self.networkActivationCosts, self.networkActivationCostsDict = \
            CignActivationCostCalculator.calculate_mac_cost(
                network=self,
                node_costs=self.nodeCosts)
        self.reachabilityMatrices = CignReachabilityMatricesCalculation.calculate_reachibility_matrices(
            network=self,
            action_spaces=self.actionSpaces)
        print("X")

    def get_max_trajectory_length(self) -> int:
        return int(self.depth - 1)

    def build_action_spaces(self):
        max_trajectory_length = self.get_max_trajectory_length()
        self.actionSpaces = []
        for t in range(max_trajectory_length):
            next_level_node_count = len(self.orderedNodesPerLevel[t + 1])
            action_count = (2 ** next_level_node_count) - 1
            action_space = []
            for action_id in range(action_count):
                action_code = action_id + 1
                l = [int(x) for x in list('{0:0b}'.format(action_code))]
                k = [0] * (next_level_node_count - len(l))
                k.extend(l)
                binary_node_selection = np.array(k)
                action_space.append(binary_node_selection)
            action_space = np.stack(action_space, axis=0)
            self.actionSpaces.append(action_space)
