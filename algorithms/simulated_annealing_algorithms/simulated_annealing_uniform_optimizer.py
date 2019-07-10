import numpy as np

from algorithms.simulated_annealing_algorithms.simulated_annealing_threshold_optimizer import \
    SimulatedAnnealingThresholdOptimizer


class SimulatedAnnealingUniformOptimizer(SimulatedAnnealingThresholdOptimizer):
    def __init__(self, network, max_num_of_iterations, annealing_schedule, balance_coefficient, use_weighted_scoring,
                 multipath_score_calculators, verbose, neighbor_volume_ratio):
        super().__init__(network, max_num_of_iterations, annealing_schedule, balance_coefficient, use_weighted_scoring,
                         multipath_score_calculators, verbose)
        self.neighborVolumeRatio = neighbor_volume_ratio

    def get_neighbor(self, threshold_state):
        new_threshold_state = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(self.network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            volume_edge = self.neighborVolumeRatio * max_threshold
            curr_thresholds = threshold_state[node.index]
            lower_bounds = np.maximum(curr_thresholds - volume_edge / 2.0, np.zeros_like(curr_thresholds))
            upper_bounds = np.minimum(curr_thresholds + volume_edge / 2.0,
                                      max_threshold * np.ones_like(curr_thresholds))
            new_thresholds = np.random.uniform(lower_bounds, upper_bounds)
            new_threshold_state[node.index] = new_thresholds
        return new_threshold_state


