import numpy as np


class ThresholdOptimizer:
    def __init__(self, run_id, network, multipath_score_calculators, balance_coefficient, use_weighted_scoring,
                 verbose):
        self.runId = run_id
        self.network = network
        self.multipathScoreCalculators = multipath_score_calculators
        self.balanceCoefficient = balance_coefficient
        self.useWeightedScoring = use_weighted_scoring
        self.verbose = verbose

    def pick_fully_random_state(self):
        threshold_state = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(self.network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            thresholds = max_threshold * np.random.uniform(size=(child_count,))
            threshold_state[node.index] = thresholds
        return threshold_state

    def calculate_threshold_score(self, threshold_state):
        scores = []
        accuracies = []
        computation_overloads = []
        for scorer in self.multipathScoreCalculators:
            res_method_0, res_method_1 = scorer.calculate_for_threshold(thresholds_dict=threshold_state)
            result = res_method_1 if self.useWeightedScoring else res_method_0
            accuracy_gain = self.balanceCoefficient * result.accuracy
            computation_overload_loss = (1.0 - self.balanceCoefficient) * (result.computationOverload - 1.0)
            score = accuracy_gain - computation_overload_loss
            accuracies.append(result.accuracy)
            computation_overloads.append(result.computationOverload)
            scores.append(score)
        final_score = np.mean(np.array(scores))
        final_accuracy = np.mean(np.array(accuracies))
        final_computation_overload = np.mean(np.array(computation_overloads))
        return final_score, final_accuracy, final_computation_overload

    def run(self):
        pass
