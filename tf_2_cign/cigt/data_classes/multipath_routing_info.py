import numpy as np
import tensorflow as tf
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
