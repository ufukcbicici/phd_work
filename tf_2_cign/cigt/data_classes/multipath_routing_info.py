import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# @dataclass
# class MultipathRouting:
#     level: int
#     curr_index: int
#     accuracy: float
#     mac: float
#     sample_count: int
#     sample_configurations: Dict[int, tuple]
#     entropy_orderings: List[List[tuple]]
from tf_2_cign.utilities.utilities import Utilities


class MultipathCombinationInfo(object):
    def __init__(self, batch_size, path_counts):
        self.batchSize = batch_size
        self.pathCounts = path_counts
        self.combinations_y_dict = {}
        self.combinations_y_hat_dict = {}
        self.combinations_routing_probabilities_dict = {}
        self.combinations_routing_entropies_dict = {}
        self.decisions_per_level = []
        for path_count in path_counts[1:]:
            decision_arrays = [[0, 1] for _ in range(path_count)]
            decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
            decision_combinations = [tpl for tpl in decision_combinations if sum(tpl) > 0]
            self.decisions_per_level.append(decision_combinations)

        self.decision_combinations_per_level = Utilities.get_cartesian_product(list_of_lists=self.decisions_per_level)
        self.decision_combinations_per_level = [tuple(np.concatenate(dc))
                                                for dc in self.decision_combinations_per_level]
        self.past_decisions_entropies_dict = {}
        self.past_decisions_routing_probabilities_dict = {}
        self.past_decisions_entropies_list = []
        self.past_decisions_routing_probabilities_list = []
        self.past_decisions_validity_array = None
        self.past_decisions_mac_array = None

    def get_total_sample_count(self):
        total_sample_count = set()
        for ll in self.combinations_routing_probabilities_dict.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for ll in self.combinations_routing_entropies_dict.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for arr in self.combinations_y_hat_dict.values():
            total_sample_count.add(arr.shape[0])
        for arr in self.combinations_y_dict.values():
            total_sample_count.add(arr.shape[0])
        assert len(total_sample_count) == 1
        total_sample_count = list(total_sample_count)[0]
        return total_sample_count

    def generate_routing_info(self, cigt, dataset):
        for decision_combination in tqdm(self.decision_combinations_per_level):
            self.add_new_combination(decision_combination=decision_combination)

            thresholds_list = np.logical_not(np.array(decision_combination)).astype(np.float)
            thresholds_list = 3.0 * thresholds_list - 1.5
            prob_thresholds_arr = np.stack([thresholds_list] * cigt.batchSize, axis=0)
            cigt.routingProbabilityThresholds.assign(prob_thresholds_arr)

            self.fill_data_buffers_for_combination(cigt=cigt, dataset=dataset,
                                                   decision_combination=decision_combination)

        past_routes = [0]
        total_sample_count = self.get_total_sample_count()
        for block_id, route_count in enumerate(cigt.pathCounts[1:]):
            # Prepare all possible valid decisions that can be taken by samples in this stage of the CIGT, based on past
            # routing decisions.
            dict_distinct_past_decisions = {}

            for combination in self.decision_combinations_per_level:
                past_route = combination[:sum(past_routes)]
                if past_route not in dict_distinct_past_decisions:
                    dict_distinct_past_decisions[past_route] = []
                dict_distinct_past_decisions[past_route].append(combination)

            for k, v in dict_distinct_past_decisions.items():
                all_entropies = np.stack([self.combinations_routing_entropies_dict[p_][block_id] for p_ in v], axis=-1)
                mean_entropies = np.mean(all_entropies, axis=-1)
                for p_ in v:
                    assert np.allclose(mean_entropies, self.combinations_routing_entropies_dict[p_][block_id])
                self.past_decisions_entropies_dict[k] = mean_entropies

                all_routing_probabilities = np.stack(
                    [self.combinations_routing_probabilities_dict[p_][block_id] for p_ in v], axis=-1)
                mean_probabilities = np.mean(all_routing_probabilities, axis=-1)
                for p_ in v:
                    assert np.allclose(mean_probabilities, self.combinations_routing_probabilities_dict[p_][block_id])
                self.past_decisions_routing_probabilities_dict[k] = mean_probabilities

            past_decisions_entropies_arr_shape = np.concatenate([
                np.array([2 for _ in range(sum(past_routes))], dtype=np.int32),
                np.array([total_sample_count], dtype=np.int32)], dtype=np.int32)
            past_decisions_entropies_arr = np.zeros(shape=past_decisions_entropies_arr_shape, dtype=np.float)
            self.past_decisions_entropies_list.append(past_decisions_entropies_arr)
            for k, v in self.past_decisions_entropies_dict.items():
                if len(k) == past_routes[block_id]:
                    self.past_decisions_entropies_list[block_id][k] = v

            past_decisions_routing_probabilities_arr_shape = np.concatenate([
                np.array([2 for _ in range(sum(past_routes))], dtype=np.int32),
                np.array([total_sample_count, route_count], dtype=np.int32)], dtype=np.int32)
            past_decisions_routing_probabilities_arr = np.zeros(shape=past_decisions_routing_probabilities_arr_shape,
                                                                dtype=np.float)
            self.past_decisions_routing_probabilities_list.append(past_decisions_routing_probabilities_arr)
            for k, v in self.past_decisions_routing_probabilities_dict.items():
                if len(k) == past_routes[block_id]:
                    self.past_decisions_routing_probabilities_list[block_id][k] = v

            past_routes.append(route_count)

        past_decisions_validity_array_shape = np.concatenate([
            np.array([2 for _ in range(sum(past_routes))], dtype=np.int32),
            np.array([total_sample_count], dtype=np.int32)], dtype=np.int32)
        self.past_decisions_validity_array = np.zeros(shape=past_decisions_validity_array_shape, dtype=np.float)
        self.past_decisions_validity_array[:] = np.nan

        for k in self.combinations_y_dict:
            y_ = self.combinations_y_dict[k]
            y_hat = np.argmax(self.combinations_y_hat_dict[k], axis=1)
            validity_vector = (y_ == y_hat).astype(np.float)
            self.past_decisions_validity_array[k] = validity_vector

        self.past_decisions_mac_array = np.zeros(shape=past_decisions_validity_array_shape[:-1], dtype=np.float)
        self.past_decisions_mac_array[:] = np.nan

        for k in self.combinations_y_dict:
            past_routes = [0]
            total_cost = cigt.cigtNodes[0].macCost
            for block_id, route_count in enumerate(cigt.pathCounts[1:]):
                single_block_cost = cigt.cigtNodes[block_id + 1].macCost
                num_of_used_nodes = k[sum(past_routes):sum(past_routes) + route_count]
                num_of_used_nodes = sum(num_of_used_nodes)
                total_cost += (num_of_used_nodes * single_block_cost)
                past_routes.append(route_count)
            self.past_decisions_mac_array[k] = total_cost
        print("X")

    def assert_routing_validity(self, cigt):
        # Assert routing probability integrities
        past_sum = 0
        for block_id, route_count in enumerate(cigt.pathCounts[1:]):
            routing_probabilities_dict = {}
            for decision_combination in tqdm(self.decision_combinations_per_level):
                past_combination = decision_combination[0:past_sum]
                if past_combination not in routing_probabilities_dict:
                    routing_probabilities_dict[past_combination] = []
                routing_probabilities_dict[past_combination].append(
                    self.combinations_routing_probabilities_dict[decision_combination][block_id])
            for k, arr in routing_probabilities_dict.items():
                for i_ in range(len(arr) - 1):
                    assert np.allclose(arr[i_], arr[i_ + 1])
            past_sum += route_count

    def add_new_combination(self, decision_combination):
        # enforced_decision_arr = np.zeros(shape=(self.batchSize, len(self.pathCounts) - 1),  dtype=np.int32)
        self.combinations_y_dict[decision_combination] = []
        self.combinations_y_hat_dict[decision_combination] = []
        self.combinations_routing_probabilities_dict[decision_combination] = []
        self.combinations_routing_entropies_dict[decision_combination] = []
        for _ in range(len(self.pathCounts) - 1):
            self.combinations_routing_probabilities_dict[decision_combination].append([])

    def add_network_outputs(self, decision_combination, y_np, logits_np, routing_probabilities_list):
        self.combinations_y_dict[decision_combination].append(y_np)
        self.combinations_y_hat_dict[decision_combination].append(logits_np)
        for block_id, arr in enumerate(routing_probabilities_list):
            self.combinations_routing_probabilities_dict[decision_combination][block_id].append(arr.numpy())

    def concat_data(self, decision_combination):
        self.combinations_y_dict[decision_combination] = np.concatenate(
            self.combinations_y_dict[decision_combination], axis=0)
        self.combinations_y_hat_dict[decision_combination] = np.concatenate(
            self.combinations_y_hat_dict[decision_combination], axis=0)

        for block_id in range(len(self.combinations_routing_probabilities_dict[decision_combination])):
            self.combinations_routing_probabilities_dict[decision_combination][block_id] = \
                np.concatenate(self.combinations_routing_probabilities_dict[decision_combination][block_id], axis=0)
            self.combinations_routing_entropies_dict[decision_combination].append(
                Utilities.calculate_entropies(self.combinations_routing_probabilities_dict[
                                                  decision_combination][block_id]))

    def fill_data_buffers_for_combination(self, cigt, dataset, decision_combination):
        for x_, y_ in dataset:
            results_dict = cigt.call(inputs=[x_, y_,
                                             tf.convert_to_tensor(1.0),
                                             tf.convert_to_tensor(False)], training=False)
            self.add_network_outputs(decision_combination=decision_combination,
                                     y_np=y_.numpy(),
                                     logits_np=results_dict["logits"].numpy(),
                                     routing_probabilities_list=results_dict["routing_probabilities"])
        self.concat_data(decision_combination=decision_combination)

    def assess_accuracy(self):
        # Assert integrity of outputs
        y_matrix = np.stack(list(self.combinations_y_dict.values()), axis=1)
        y_avg = np.mean(y_matrix, axis=1).astype(dtype=y_matrix.dtype)
        y_diff = y_matrix - y_avg[:, np.newaxis]
        assert np.all(y_diff == 0)
        y_hat_matrix = np.stack(list(self.combinations_y_hat_dict.values()), axis=2)
        y_hat_matrix = np.argmax(y_hat_matrix, axis=1)
        equals_matrix = np.equal(y_hat_matrix, y_avg[:, np.newaxis])
        correct_vec = np.sum(equals_matrix, axis=1)
        best_accuracy = np.mean(correct_vec > 0.0)
        print("best_accuracy={0}".format(best_accuracy))

    def get_default_accuracy(self, cigt, indices):
        routing_decisions_arr = np.zeros(shape=(len(indices), sum(cigt.pathCounts[1:])), dtype=np.int32)
        curr_index = 0
        for block_id in range(len(cigt.pathCounts) - 1):
            routing_decisions_so_far = routing_decisions_arr[:, :curr_index]
            index_arrays = [routing_decisions_so_far[:, col] for col in range(routing_decisions_so_far.shape[1])]
            index_arrays.append(indices)
            routing_probabilities_for_block = self.past_decisions_routing_probabilities_list[block_id][index_arrays]
            print("X")

        # for block_id in range(len(cigt.pathCounts) - 1):
        #     self.combinations_routing_probabilities_dict


    # list_of_entropy_intervals: length is equal to the number of decision blocks. Each i. element of the list
    # is a numpy array, which has the shape (number_of_entropy_intervals[i], 2).

    # list_of_probability_thresholds: length is equal to the number of decision blocks. Each i. element of the list
    # is a numpy array, which has the shape (number_of_entropy_intervals[i], route_count).

    # indices: Indices to be used for this optimization calculation..

