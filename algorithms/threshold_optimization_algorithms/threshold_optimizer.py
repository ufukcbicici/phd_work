import numpy as np


class ThresholdOptimizer:
    def __init__(self, run_id, network, routing_data_dict, multipath_score_calculators, balance_coefficient,
                 test_ratio,
                 use_weighted_scoring,
                 verbose):
        self.runId = run_id
        self.network = network
        self.testRatio = test_ratio
        self.routingDataDict = routing_data_dict
        self.multipathScoreCalculators = multipath_score_calculators
        self.iterations = str(list(self.multipathScoreCalculators.keys()))
        self.balanceCoefficient = balance_coefficient
        self.testRatio = test_ratio
        self.useWeightedScoring = use_weighted_scoring
        self.verbose = verbose
        self.validationDataDict = {}
        self.testDataDict = {}
        for iteration, routing_data in self.routingDataDict.items():
            self.validationDataDict[iteration], self.testDataDict[iteration] = \
                routing_data.apply_validation_test_split(test_ratio=self.testRatio)

    def thresholds_to_numpy(self, thresholds):
        sorted_indices = sorted([node.index for node in self.network.topologicalSortedNodes if not node.isLeaf])
        arr_list = []
        for threshold_dict in thresholds:
            assert len(threshold_dict) == len(sorted_indices)
            arr_as_list = [threshold_dict[node_idx] for node_idx in sorted_indices]
            threshold_arr = np.concatenate(arr_as_list)
            arr_list.append(threshold_arr)
        thresholds_matrix = np.stack(arr_list, axis=0)
        return thresholds_matrix

    def numpy_to_threshold_dict(self, thresholds_arr):
        child_counts = {node.index: len(self.network.dagObject.children(node=node)) for
                        node in self.network.topologicalSortedNodes
                        if not node.isLeaf}
        threshold_dict = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            other_children_counts = sum([v for k, v in child_counts.items() if k < node.index])
            this_children_count = child_counts[node.index]
            threshold_dict[node.index] = thresholds_arr[other_children_counts:
                                                        other_children_counts + this_children_count]
        assert sum(child_counts.values()) == sum([len(v) for v in threshold_dict.values()])
        return threshold_dict

    def pick_fully_random_state(self, as_numpy=False):
        threshold_state = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(self.network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            thresholds = max_threshold * np.random.uniform(size=(child_count,))
            threshold_state[node.index] = thresholds
        if as_numpy:
            return self.thresholds_to_numpy(thresholds=[threshold_state])
        return threshold_state

    def calculate_threshold_score(self, threshold_state, routing_data_dict):
        scores = []
        accuracies = []
        computation_overloads = []
        for iteration, scorer in self.multipathScoreCalculators.items():
            routing_data = routing_data_dict[iteration]
            res_method_0, res_method_1 = scorer.calculate_for_threshold(thresholds_dict=threshold_state,
                                                                        routing_data=routing_data)
            result = res_method_1 if self.useWeightedScoring else res_method_0
            accuracy_gain = self.balanceCoefficient * result.accuracy
            computation_overload_loss = (1.0 - self.balanceCoefficient) * result.computationOverload
            score = accuracy_gain - computation_overload_loss
            accuracies.append(result.accuracy)
            computation_overloads.append(result.computationOverload)
            scores.append(score)
        final_score = np.mean(np.array(scores))
        final_accuracy = np.mean(np.array(accuracies))
        final_computation_overload = np.mean(np.array(computation_overloads))
        return final_score, final_accuracy, final_computation_overload, result

    def run(self):
        pass
