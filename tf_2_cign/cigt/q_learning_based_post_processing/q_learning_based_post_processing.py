import os
import numpy as np

from sklearn.model_selection import train_test_split

from tf_2_cign.cigt.data_classes.multipath_routing_info2 import MultipathCombinationInfo2


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

