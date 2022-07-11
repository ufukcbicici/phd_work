import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from auxillary.db_logger import DbLogger
from tf_2_cign.cigt.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer

# Hyper-parameters
from tf_2_cign.utilities.utilities import Utilities


@dataclass
class SearchStatus:
    level: int
    curr_index: int
    accuracy: float
    mac: float
    sample_count: int
    sample_configurations: Dict[int, tuple]
    entropy_orderings: List[List[tuple]]


class MultipathThresholdOptimizer(BayesianOptimizer):
    def __init__(self, xi, init_points, n_iter, accuracy_mac_balance_coeff,
                 model_id, val_ratio):
        super().__init__(xi, init_points, n_iter)
        self.model = None
        self.modelId = model_id
        self.valRatio = val_ratio
        self.accuracyMacBalanceCoeff = accuracy_mac_balance_coeff
        self.routingProbabilities, self.routingEntropies, self.logits, self.groundTruths, \
        self.fullTrainingAccuracy, self.fullTestAccuracy = self.get_model_outputs()
        probabilities_arr = list(self.routingProbabilities.values())[0]
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        self.routingBlocksCount = 0
        for idx, arr in enumerate(probabilities_arr):
            self.routingBlocksCount += 1
            num_of_routes = arr.shape[1]
            self.maxEntropies.append(-np.log(1.0 / num_of_routes))
            self.optimization_bounds_continuous["entropy_block_{0}".format(idx)] = (0.0, self.maxEntropies[idx])
        self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()
        self.listOfEntropiesPerLevel, self.fullEntropyArray = self.prepare_entropies_per_level_and_decision()
        self.routingCorrectnessDict, self.routingMacDict, self.valBaseAccuracy, \
            self.testBaseAccuracy, self.fullAccuracyArray = self.get_correctness_and_mac_dicts()
        self.reformat_arrays()
        self.runId = DbLogger.get_run_id()
        kv_rows = [(self.runId, "Validation Sample Count", "{0}".format(len(self.valIndices))),
                   (self.runId, "Test Sample Count", "{0}".format(len(self.testIndices))),
                   (self.runId, "Validation Base Accuracy", "{0}".format(self.valBaseAccuracy)),
                   (self.runId, "Test Base Accuracy", "{0}".format(self.testBaseAccuracy)),
                   (self.runId, "Full Test Accuracy", "{0}".format(self.fullTestAccuracy)),
                   (self.runId, "xi", "{0}".format(self.xi)),
                   (self.runId, "init_points", "{0}".format(self.init_points)),
                   (self.runId, "n_iter", "{0}".format(self.n_iter)),
                   (self.runId, "accuracyMacBalanceCoeff", "{0}".format(self.accuracyMacBalanceCoeff)),
                   (self.runId, "modelId", "{0}".format(self.modelId)),
                   (self.runId, "Base Mac", "{0}".format(self.routingMacDict[(0, 0)]))
                   ]
        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")

    # Calculate entropies per level and per decision. The list by itself represents the levels.
    # Each element of the list is a numpy array, whose second and larger dimensions represent the
    # decisions taken in previous levels.
    def prepare_entropies_per_level_and_decision(self):
        decision_arrays = [[0, 1] for _ in range(self.routingBlocksCount)]
        decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        list_of_entropies_per_level = []
        ent_arr_shape = (self.totalSampleCount, *[2 for _ in range(self.routingBlocksCount)], self.routingBlocksCount)
        full_entropy_array = np.zeros(shape=ent_arr_shape, dtype=np.float)

        for block_id in range(self.routingBlocksCount):
            num_of_decision_dimensions = 2 ** block_id
            entropy_array = np.zeros(shape=(self.totalSampleCount, num_of_decision_dimensions))
            list_of_entropies_per_level.append(entropy_array)

            all_previous_combinations = Utilities.get_cartesian_product(
                [[0, 1] for _ in range(block_id)])
            for previous_combination in all_previous_combinations:
                valid_combinations = []
                for combination in decision_combinations:
                    if combination[0:block_id] == previous_combination:
                        valid_combinations.append(combination)
                valid_probability_arrays = []
                valid_entropy_arrays = []
                for valid_combination in valid_combinations:
                    probability_array = self.routingProbabilities[valid_combination][block_id]
                    entropy_array = self.routingEntropies[valid_combination][block_id]
                    valid_probability_arrays.append(probability_array)
                    valid_entropy_arrays.append(entropy_array)

                # Assert that the result of same action sequences are always equal.
                for i_ in range(len(valid_probability_arrays) - 1):
                    assert np.allclose(valid_probability_arrays[i_], valid_probability_arrays[i_ + 1])
                for i_ in range(len(valid_entropy_arrays) - 1):
                    assert np.allclose(valid_entropy_arrays[i_], valid_entropy_arrays[i_ + 1])

                valid_entropies_matrix = np.stack(valid_entropy_arrays, axis=1)
                valid_entropy_arr = np.mean(valid_entropies_matrix, axis=1)
                if len(all_previous_combinations) == 1:
                    assert all_previous_combinations[0] == ()
                    list_of_entropies_per_level[block_id][:, 0] = valid_entropy_arr
                else:
                    combination_coord = int("".join(str(ele) for ele in previous_combination), 2)
                    list_of_entropies_per_level[block_id][:, combination_coord] = valid_entropy_arr

        # for sample_id in range(self.totalSampleCount):
        for combination in decision_combinations:
            for block_id in range(self.routingBlocksCount):
                e_array = self.routingEntropies[combination][block_id]
                index_arrays = [list(range(self.totalSampleCount))]
                for i_ in combination:
                    index_arrays.append([i_] * self.totalSampleCount)
                index_arrays.append([block_id] * self.totalSampleCount)
                full_entropy_array[index_arrays] = e_array
        return list_of_entropies_per_level, full_entropy_array

    def get_correctness_and_mac_dicts(self):
        acc_arr_shape = (self.totalSampleCount, *[2 for _ in range(self.routingBlocksCount)])
        full_accuracy_array = np.zeros(shape=acc_arr_shape, dtype=np.float)
        correctness_dict = {}
        mac_dict = {}
        decision_arrays = [[0, 1] for _ in range(self.routingBlocksCount)]
        decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        for decision_combination in decision_combinations:
            correctness_dict[decision_combination] = []
            # Get the mac cost for this routing combination.
            combination_mac_cost = self.model.cigtNodes[0].macCost
            for idx, decision in enumerate(decision_combination):
                level_mac_cost = self.model.cigtNodes[idx + 1].macCost
                if decision == 0:
                    combination_mac_cost += level_mac_cost
                else:
                    combination_mac_cost += self.model.pathCounts[idx + 1] * level_mac_cost
            mac_dict[decision_combination] = combination_mac_cost

            # Get correctness vectors.
            for idx in range(self.totalSampleCount):
                correct_label = self.groundTruths[decision_combination][idx]
                logits = self.logits[decision_combination][idx]
                estimated_label = np.argmax(logits)
                correctness_dict[decision_combination].append(int(correct_label == estimated_label))
            index_arrays = [list(range(self.totalSampleCount))]
            for i_ in decision_combination:
                index_arrays.append([i_] * self.totalSampleCount)
            full_accuracy_array[index_arrays] = correctness_dict[decision_combination]

        val_ground_truths = self.groundTruths[(0, 0)][self.valIndices]
        val_logits = self.logits[(0, 0)][self.valIndices]
        val_estimated_labels = np.argmax(val_logits, axis=1)
        val_base_accuracy = np.mean(val_ground_truths == val_estimated_labels)

        test_ground_truths = self.groundTruths[(0, 0)][self.testIndices]
        test_logits = self.logits[(0, 0)][self.testIndices]
        test_estimated_labels = np.argmax(test_logits, axis=1)
        test_base_accuracy = np.mean(test_ground_truths == test_estimated_labels)

        return correctness_dict, mac_dict, val_base_accuracy, test_base_accuracy, full_accuracy_array

    def reformat_arrays(self):
        # Accuracy array
        acc_arr_shape = (self.totalSampleCount, *[2 for _ in range(self.routingBlocksCount)])
        accuracy_array = np.zeros(shape=acc_arr_shape, dtype=np.float)
        # Entropy array
        ent_arr_shape = (self.totalSampleCount, *[2 for _ in range(self.routingBlocksCount)], self.routingBlocksCount)
        entropy_array = np.zeros(shape=ent_arr_shape, dtype=np.float)
        print("X")

        # for ll in range(self.routingBlocksCount):

    def prepare_val_test_sets(self):
        total_sample_count = set()
        for ll in self.routingProbabilities.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for ll in self.routingEntropies.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for arr in self.logits.values():
            total_sample_count.add(arr.shape[0])
        for arr in self.groundTruths.values():
            total_sample_count.add(arr.shape[0])
        assert len(total_sample_count) == 1
        total_sample_count = list(total_sample_count)[0]
        val_sample_count = int(total_sample_count * self.valRatio)
        indices = np.arange(total_sample_count)
        val_indices, test_indices = train_test_split(indices, train_size=val_sample_count)
        return total_sample_count, val_indices, test_indices

    def get_model_outputs(self):
        return {}, {}, {}, {}, 0.0, 0.0

    def get_metrics(self, indices, thresholds):
        selections_arr = np.zeros(shape=(len(indices), self.routingBlocksCount), dtype=np.int32)
        selections_arr[:] = -10000
        for level in range(self.routingBlocksCount):
            threshold = thresholds[level]
            if level == 0:
                curr_entropies = self.listOfEntropiesPerLevel[level][indices][:, 0]
            else:
                selection_coords = np.apply_along_axis(func1d=lambda r: int("".join(str(ele) for ele in r), 2),
                                                       axis=1, arr=selections_arr[:, 0:level])
                curr_entropies = self.listOfEntropiesPerLevel[level][indices][np.arange(len(indices)), selection_coords]
            this_level_selections = np.array(curr_entropies >= threshold, dtype=np.int32)
            selections_arr[:, level] = this_level_selections
        # Get accuracy and mac results
        correct_list = []
        mac_ratio_list = []
        smallest_mac = np.min(list(self.routingMacDict.values()))
        for ii, si in enumerate(indices):
            selection_trajectory = tuple(selections_arr[ii, :])
            is_correct = self.routingCorrectnessDict[selection_trajectory][si]
            correct_list.append(is_correct)
            selection_mac_ratio = self.routingMacDict[selection_trajectory] / smallest_mac
            mac_ratio_list.append(selection_mac_ratio)
        accuracy = np.mean(correct_list)
        avg_mac_ratio = np.mean(mac_ratio_list)
        accuracy_component = self.accuracyMacBalanceCoeff * accuracy
        mac_component = (1.0 - self.accuracyMacBalanceCoeff) * (avg_mac_ratio - 1.0)
        score = accuracy_component - mac_component
        return np.asscalar(score), np.asscalar(accuracy), np.asscalar(avg_mac_ratio)

    def cost_function(self, **kwargs):
        thresholds = []
        for level in range(self.routingBlocksCount):
            thresholds.append(kwargs["entropy_block_{0}".format(level)])

        val_score, val_accuracy, val_mac_overload_percentage = \
            self.get_metrics(indices=self.valIndices, thresholds=thresholds)
        test_score, test_accuracy, test_mac_overload_percentage = \
            self.get_metrics(indices=self.testIndices, thresholds=thresholds)
        print("************************************************************************************************")
        print("val_score:{0} val_accuracy:{1} val_mac_overload_percentage:{2}".format(val_score, val_accuracy,
                                                                                      val_mac_overload_percentage))
        print("test_score:{0} test_accuracy:{1} test_mac_overload_percentage:{2}".format(test_score, test_accuracy,
                                                                                         test_mac_overload_percentage))
        print("************************************************************************************************")
        return val_score

    def change_statistics_for_single_sample(self, indices, route_array, entropy_array, level, sample_index):
        # base_mac = self.routingMacDict[(0, 0)]
        # curr_valid_count = len(indices) * acc
        # curr_total_mac = len(indices) * mac

        # Set all routes for higher levels to zero
        if level < route_array.shape[1] - 1:
            route_array[:, level:] = 0
        # Set the level decision for this sample to one.
        assert route_array[sample_index, level] == 0
        route_array[sample_index, level] = 1

        # curr_sample_route = route_array[sample_index, :]
        # assert curr_sample_route[level] == 0
        # old_route = tuple(curr_sample_route)
        # curr_sample_route[level] = 1
        # new_route = tuple(curr_sample_route)

        # # Get accuracy and mac wrt old route
        # old_accuracy = self.routingCorrectnessDict[old_route][sample_index]
        # old_sample_mac = self.routingMacDict[old_route] / base_mac
        # # Get accuracy and mac wrt new route
        # new_accuracy = self.routingCorrectnessDict[new_route][sample_index]
        # new_sample_mac = self.routingMacDict[new_route] / base_mac
        # new_valid_count = curr_valid_count - old_accuracy + new_accuracy
        # new_total_mac = curr_total_mac - old_sample_mac + new_sample_mac
        # new_acc = new_valid_count / len(indices)
        # new_mac = new_total_mac / len(indices)

        # Re-sort following levels according to changing entropies (for levels smaller than route_array.shape[1] - 1)

        #
        #
        return 0, 0

    def run_brute_force(self, first_block_index, indices):
        set_of_accuracies_macs = set()
        route_array = np.zeros(shape=(len(indices), self.routingBlocksCount), dtype=np.int32)
        entropy_array = []
        for jdx in range(self.routingBlocksCount):
            entropy_arr = self.routingEntropies[(0, 0)][jdx][indices]
            entropy_arr_with_indices = [tpl for tpl in zip(indices, entropy_arr)]
            entropy_arr_with_indices = sorted(entropy_arr_with_indices, key=lambda tpl: tpl[1])
            entropy_arr_with_indices = np.array(entropy_arr_with_indices, dtype=[("sample_index", int),
                                                                                 ("entropy", float)])
            entropy_array.append(entropy_arr_with_indices)
        entropy_array = np.stack(entropy_array, axis=0)
        #
        # # Base statistics when every sample just follows a single path
        # curr_score, curr_accuracy, curr_mac = self.get_metrics(indices=indices, thresholds=[np.inf, np.inf])
        # # set_of_accuracies_macs.add((curr_accuracy, curr_mac))
        #
        # # Set the decisions of the first block first.
        # for fbi in range(first_block_index):
        #     tpl = entropy_array[0][fbi]
        #     sample_index = tpl[0]
        #     curr_accuracy, curr_mac = self.change_statistics_for_single_sample(indices=indices,
        #                                                                        route_array=route_array,
        #                                                                        entropy_array=entropy_array,
        #                                                                        level=0,
        #                                                                        sample_index=sample_index)
        # set_of_accuracies_macs.add((curr_accuracy, curr_mac))
        #
        # # Start the search
        # combs = [list(range(len(indices))) for _ in range(self.routingBlocksCount - 1)]
        # combs = Utilities.get_cartesian_product(list_of_lists=combs)
        # for comb in combs:
        #     comb

        # Prepare status data for each level
        # for level in range(self.routingBlocksCount):
        #     if level == 0:
        #         route_a

        # decision_arrays = [list(range(len(indices))) for _ in range(self.routingBlocksCount - 1)]
        # decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        # sample_configurations = {}
        # for ii in indices:
        #     sample_configurations[ii] = tuple([0 for _ in range(self.routingBlocksCount)])

        # Entropy ordering

    def apply_brute_force_solution(self, indices):
        if indices is None:
            indices = np.arange(self.totalSampleCount)
        # # Step 0: Enumerate all possible configurations. A configuration (i_0, i_1, ..., i_(n-1)) means
        # # when we have the scale of entropies at each level, we have just passed the i_0. bin at level 0, i_1. bin
        # # at level 1 and etc. Note that when step to the next bin at level i_j, all bins at higher levels (j+1, j+2,
        # # ..., n-1) do change.
        # decision_arrays = [list(range(len(indices))) for _ in range(self.routingBlocksCount - 1)]
        # decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        # # Step 1: A search status for every level
        # search_status_list = []
        # for idx in range(self.routingBlocksCount):
        #     score, accuracy, mac = self.get_metrics(indices=indices, thresholds=[np.inf, np.inf])
        #     score = np.asscalar(score)
        #     accuracy = np.asscalar(accuracy)
        #     mac = np.asscalar(mac)
        #
        #
        #
        #
        #
        #     search_status = SearchStatus(accuracy=accuracy, mac=mac, sample_count=len(indices),
        #                                  sample_configurations=sample_configurations,
        #                                  entropy_orderings=entropy_orderings)
        #     search_status_list.append(search_status)

        for outer_index in range(len(indices)):
            self.run_brute_force(first_block_index=outer_index, indices=indices)
            # for inner_combination in decision_combinations:
            #     routing_combination = tuple([outer_index, *inner_combination])

            #     print("X")
            #
            # print("X")

# @dataclass
# class SearchStatus:
#     accuracy: float
#     mac: float
#     sample_count: int
#     sample_configurations: List[List[tuple]]
