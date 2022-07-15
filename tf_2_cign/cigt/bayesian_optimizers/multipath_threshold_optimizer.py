import numpy as np
import itertools
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

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
        self.routingProbabilities, self.routingEntropies, self.logits, self.groundTruths, self. \
            fullTrainingAccuracy, self.fullTestAccuracy = self.get_model_outputs()
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
        self.routingCorrectnessDict, self.routingMacDict, self.valBaseAccuracy, self.testBaseAccuracy, self. \
            fullAccuracyArray, self.fullMacArray = self.get_correctness_and_mac_dicts()
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
        full_mac_array = np.zeros(shape=tuple([2 for _ in range(self.routingBlocksCount)]), dtype=np.float)
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

        for tpl, v in mac_dict.items():
            full_mac_array[tpl] = v

        return correctness_dict, mac_dict, val_base_accuracy, test_base_accuracy, full_accuracy_array, full_mac_array

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

    def get_metrics_v2(self, indices, thresholds):
        # Get routing selections
        selections_arr = []
        for level in range(self.routingBlocksCount):
            threshold = thresholds[level]
            index_array = [indices]
            for arr in selections_arr:
                index_array.append(arr)
            for _ in range(self.routingBlocksCount - level):
                index_array.append(np.array([0] * len(indices)))
            index_array.append(np.array([level] * len(indices)))
            assert len(index_array) == len(self.fullEntropyArray.shape)
            curr_entropies = self.fullEntropyArray[index_array]
            this_level_selections = np.array(curr_entropies >= threshold, dtype=np.int32)
            selections_arr.append(this_level_selections)

        # Get accuracy selections
        index_array = [indices]
        index_array.extend(selections_arr)
        accuracy_arr = self.fullAccuracyArray[index_array]
        accuracy = np.mean(accuracy_arr)

        # Get mac selections
        mac_array = self.fullMacArray[selections_arr] * (1.0 / self.fullMacArray[0, 0])
        average_mac = np.mean(mac_array)
        return accuracy, average_mac, accuracy_arr, mac_array

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

    def work_on_threshold_batch(self, run_id, original_run_id, indices, threshold_combinations):
        rows = []
        for threshold_combination in tqdm(threshold_combinations):
            affected_indices = threshold_combination[-1][1]
            curr_thresholds = []
            curr_thresholds.extend(threshold_combination[:-1])
            curr_thresholds.append(threshold_combination[-1][0])
            curr_thresholds = tuple(curr_thresholds)
            accuracy, mean_mac, _, _ = self.get_metrics_v2(indices=indices, thresholds=curr_thresholds)
            row = [run_id, original_run_id]
            row.extend(curr_thresholds)
            row.extend([-1.0 for _ in range(4 - self.routingBlocksCount)])
            row.extend([accuracy, mean_mac])
            rows.append(row)
        return rows

    def work_on_threshold_batch_v2(self, run_id, original_run_id, indices, threshold_combinations):
        rows = []
        current_accuracy_array = np.zeros(shape=(self.totalSampleCount,), dtype=np.int32)
        current_mac_array = np.zeros(shape=(self.totalSampleCount,), dtype=np.float)
        last_combination = None
        accuracy = -1
        mean_mac = -1
        for threshold_id, threshold_combination in tqdm(enumerate(threshold_combinations)):
            affected_indices = threshold_combination[-1][1]
            curr_thresholds = []
            curr_thresholds.extend(threshold_combination[:-1])
            curr_thresholds.append(threshold_combination[-1][0])
            curr_thresholds = tuple(curr_thresholds)
            # A change in upper levels
            if len(affected_indices) == 0:
                if last_combination is not None:
                    active_block_id = np.asscalar(np.argmax(np.equal(threshold_combination, last_combination) == False))
                    assert active_block_id < self.routingBlocksCount - 1
                accuracy, mean_mac, accuracy_arr, mac_array = self.get_metrics_v2(indices=indices,
                                                                                  thresholds=curr_thresholds)
                current_accuracy_array[indices] = accuracy_arr
                current_mac_array[indices] = mac_array

            # A change in the last level
            else:
                # Trick: We know that only the samples with the indices in the list "affected_indices" can be
                # affected by the threshold level chance in the last level. We will only calculate the change in the
                # accuracy caused by the affected indices, gaining speed.
                accuracy_partial, mac_partial, new_accuracies, new_macs = self.get_metrics_v2(
                    indices=affected_indices, thresholds=curr_thresholds)
                total_valid_count = accuracy * len(indices)
                total_mac_count = mean_mac * len(indices)

                old_accuracies = current_accuracy_array[affected_indices]
                old_macs = current_mac_array[affected_indices]

                total_valid_count -= np.sum(old_accuracies)
                total_mac_count -= np.sum(old_macs)

                current_accuracy_array[affected_indices] = new_accuracies
                current_mac_array[affected_indices] = new_macs

                total_valid_count += np.sum(new_accuracies)
                total_mac_count += np.sum(new_macs)

                accuracy = total_valid_count / len(indices)
                mean_mac = total_mac_count / len(indices)
            last_combination = curr_thresholds

            row = [run_id, original_run_id]
            row.extend(curr_thresholds)
            row.extend([-1.0 for _ in range(4 - self.routingBlocksCount)])
            row.extend([accuracy, mean_mac])
            rows.append(row)

        return rows

    def apply_brute_force_solution(self, indices):
        if indices is None:
            indices = np.arange(self.totalSampleCount)

        entropies_per_level = []
        entropy_thresholds_per_level = []

        # Get all possible entropies, for every combination
        decision_arrays = [[0, 1] for _ in range(self.routingBlocksCount)]
        decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        last_level_entropies_to_samples_dict = {}
        for level in range(self.routingBlocksCount):
            entropies_per_level.append([])
            for decision in decision_combinations:
                entropies = self.routingEntropies[decision][level][indices]
                entropies_per_level[level].append(entropies)
                if level == self.routingBlocksCount - 1:
                    for ent, idx in zip(entropies, indices):
                        if ent not in last_level_entropies_to_samples_dict:
                            last_level_entropies_to_samples_dict[ent] = set()
                        last_level_entropies_to_samples_dict[ent].add(idx)
            entropies_all_combinations = np.concatenate(entropies_per_level[level])
            entropies_set = set(entropies_all_combinations)
            if level == self.routingBlocksCount - 1:
                assert len(entropies_set) == len(last_level_entropies_to_samples_dict)
            entropies_set.add(max(entropies_set) * 10.0)
            entropies_set.add(-1.0)
            assert len(entropies_set) <= (2 ** level) * len(indices)
            entropies_ordered = sorted(list(entropies_set), reverse=True)
            entropies_per_level[level] = entropies_ordered
            thresholds_arr = []
            for idx in range(len(entropies_per_level[level]) - 1):
                e0 = entropies_per_level[level][idx]
                e1 = entropies_per_level[level][idx + 1]
                thresh = 0.5 * (e0 - e1) + e1
                if level == self.routingBlocksCount - 1:
                    if idx > 0:
                        assert e0 in last_level_entropies_to_samples_dict
                        affected_samples = last_level_entropies_to_samples_dict[e0]
                        thresholds_arr.append((thresh, np.array(list(affected_samples))))
                    else:
                        thresholds_arr.append((thresh, np.array([])))
                else:
                    thresholds_arr.append(thresh)
            entropy_thresholds_per_level.append(thresholds_arr)

        # Check the correctness of the entropy arrays
        for level in range(self.routingBlocksCount):
            all_previous_combinations = Utilities.get_cartesian_product(
                [[0, 1] for _ in range(level)])
            for previous_combination in all_previous_combinations:
                valid_combinations = []
                for combination in decision_combinations:
                    if combination[0:level] == previous_combination:
                        valid_combinations.append(combination)
                arrs = []
                for valid_combination in valid_combinations:
                    index_arrays = [indices]
                    for c in valid_combination:
                        index_arrays.append(np.array([c] * len(indices)))
                    index_arrays.append(np.array([level] * len(indices)))
                    arrs.append(self.fullEntropyArray[index_arrays])
                for idx in range(len(arrs) - 1):
                    assert np.allclose(arrs[idx], arrs[idx + 1])

            if level == self.routingBlocksCount - 1:
                assert len([tpl for tpl in entropy_thresholds_per_level[level] if len(tpl[1]) == 0]) == 1

        # threshold_combinations = Utilities.get_cartesian_product(list_of_lists=entropy_thresholds_per_level)
        DbLogger.write_into_table(rows=[(self.runId, "Fast Search")], table=DbLogger.runMetaData)
        threshold_tpls = []
        cartesian_gen = itertools.product(*entropy_thresholds_per_level)
        for tpl in cartesian_gen:
            threshold_tpls.append(tpl)
            if len(threshold_tpls) >= 1000000:
                rows = self.work_on_threshold_batch_v2(run_id=self.runId,
                                                       original_run_id=self.modelId, indices=indices,
                                                       threshold_combinations=threshold_tpls)
                threshold_tpls = []
                DbLogger.write_into_table(rows=rows, table="threshold_search_2")
        if len(threshold_tpls) > 0:
            rows = self.work_on_threshold_batch_v2(run_id=self.runId,
                                                   original_run_id=self.modelId, indices=indices,
                                                   threshold_combinations=threshold_tpls)
            DbLogger.write_into_table(rows=rows, table="threshold_search_2")

        print("X")
