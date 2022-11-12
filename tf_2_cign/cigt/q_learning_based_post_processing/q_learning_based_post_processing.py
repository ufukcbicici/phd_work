import os
import numpy as np

from sklearn.model_selection import train_test_split

from tf_2_cign.cigt.data_classes.multipath_routing_info2 import MultipathCombinationInfo2
from tf_2_cign.utilities.utilities import Utilities


class QLearningRoutingOptimizer(object):
    intermediate_outputs_path = os.path.join(os.path.dirname(__file__), "..", "intermediate_outputs")

    def __init__(self, run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader,
                 model_id, val_ratio, random_seed):
        self.runId = run_id
        self.modelId = model_id
        self.valRatio = val_ratio
        self.numOfEpochs = num_of_epochs
        self.randomSeed = random_seed
        self.accuracyWeight = accuracy_weight
        self.macWeight = mac_weight
        self.modelLoader = model_loader
        self.model, self.dataset = self.modelLoader.get_model(model_id=self.modelId)
        self.pathCounts = list(self.model.pathCounts)
        self.totalSampleCount, self.valIndices, self.testIndices = None, None, None
        self.multiPathInfoObject = self.load_multipath_info()
        # self.softmaxTemperatureOptimizer = SoftmaxTemperatureOptimizer(multi_path_object=self.multiPathInfoObject)
        # self.softmaxTemperatureOptimizer.plot_entropy_histogram_with_temperature(temperature=1.0, block_id=0)
        self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()

    # Load routing information for the particular model
    def load_multipath_info(self):
        # object_path = os.path.join(object_folder_path, "multipath_info_object.pkl")
        multipath_info_object = MultipathCombinationInfo2(batch_size=self.model.batchSize,
                                                          path_counts=self.pathCounts)
        multipath_info_object.generate_routing_info(
            cigt=self.model,
            dataset=self.dataset.testDataTf,
            apply_temperature_optimization_to_entropies=False,
            apply_temperature_optimization_to_routing_probabilities=False)
        multipath_info_object.assert_routing_validity(cigt=self.model)
        multipath_info_object.assess_accuracy()
        return multipath_info_object

    def prepare_val_test_sets(self):
        total_sample_count = set()
        for ll in self.multiPathInfoObject.combinations_routing_probabilities_dict.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for ll in self.multiPathInfoObject.combinations_routing_entropies_dict.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for arr in self.multiPathInfoObject.combinations_y_hat_dict.values():
            total_sample_count.add(arr.shape[0])
        for arr in self.multiPathInfoObject.combinations_y_dict.values():
            total_sample_count.add(arr.shape[0])
        assert len(total_sample_count) == 1
        total_sample_count = list(total_sample_count)[0]
        val_sample_count = int(total_sample_count * self.valRatio)
        indices = np.arange(total_sample_count)
        np.random.seed(self.randomSeed)
        val_indices, test_indices = train_test_split(indices, train_size=val_sample_count)
        return total_sample_count, val_indices, test_indices

    def get_last_block_routing_decisions(self, q_choice_combination):
        # Determine the corresponding routing choices for every sample
        path_selections = np.arange(self.totalSampleCount)[:, np.newaxis]
        for block_id, block_choice in enumerate(q_choice_combination):
            path_indices = tuple([path_selections[:, idx] for idx in range(path_selections.shape[1])])
            routing_probabilities = \
                self.multiPathInfoObject.past_decisions_routing_probabilities_list[block_id][path_indices]
            # Select the possible routing options
            # If block_choice = 0 -> ig selection
            ig_routings = Utilities.one_hot_numpy(arr=routing_probabilities)
            # If block_choice = 1 -> all paths
            all_paths_routings = np.ones(shape=ig_routings.shape, dtype=np.int32)
            selected_paths_for_block = np.where(block_choice, all_paths_routings, ig_routings)
            path_selections = np.concatenate([path_selections[:, :-1],
                                              selected_paths_for_block,
                                              np.arange(self.totalSampleCount)[:, np.newaxis]], axis=1)
        # Accuracy vector of selections
        path_indices = tuple([path_selections[:, idx] for idx in range(path_selections.shape[1])])
        validity_vector = self.multiPathInfoObject.past_decisions_validity_array[path_indices]
        mac_vector = self.multiPathInfoObject.past_decisions_mac_array[path_indices[:-1]]
        q_values = self.accuracyWeight * validity_vector.astype(np.float) + self.macWeight * mac_vector
        return q_values

    def prepare_q_tables(self):
        q_tables = [np.zeros(shape=1)] * (len(self.model.pathCounts) - 1)
        for block_id in range(len(self.model.pathCounts) - 2, -1, -1):
            # Last block
            if block_id == len(self.model.pathCounts) - 2:
                choice_count = block_id + 1
                q_table_shape = np.concatenate([
                    np.array([2 for _ in range(choice_count)], dtype=np.int32),
                    np.array([self.totalSampleCount], dtype=np.int32)]).astype(dtype=np.int32)
                q_table = np.zeros(shape=q_table_shape, dtype=np.float32)
                choice_combinations = Utilities.get_cartesian_product(
                    list_of_lists=[[0, 1] for _ in range(choice_count)])
                for choice_combination in choice_combinations:
                    q_values = self.get_last_block_routing_decisions(q_choice_combination=choice_combination)
                    q_table[choice_combination] = q_values
                q_tables[block_id] = q_table
            else:
                q_table = np.max(q_tables[block_id + 1], axis=-2)
                q_tables[block_id] = q_table
            print(block_id)
