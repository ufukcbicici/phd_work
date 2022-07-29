import time

import numpy as np
from auxillary.db_logger import DbLogger
from tf_2_cign.cigt.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from sklearn.model_selection import train_test_split


class CrossEntropySearchOptimizer(BayesianOptimizer):
    def __init__(self, xi, init_points, n_iter, accuracy_mac_balance_coeff,
                 model_id, val_ratio):
        super().__init__(xi, init_points, n_iter)
        self.modelId = model_id
        self.valRatio = val_ratio
        self.accuracyMacBalanceCoeff = accuracy_mac_balance_coeff
        self.dataset = self.get_dataset()
        self.model = self.get_model(model_id=424)
        self.measure_model_accuracy(model=self.model, dataset=self.dataset)
        self.multiPathInfoObject = MultipathCombinationInfo(batch_size=self.model.batchSize,
                                                            path_counts=self.model.pathCounts)
        self.multiPathInfoObject.generate_routing_info(cigt=self.model, dataset=self.dataset.testDataTf)
        self.multiPathInfoObject.assert_routing_validity(cigt=self.model)
        self.multiPathInfoObject.assess_accuracy()
        self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        self.routingBlocksCount = 0
        for idx, arr in enumerate(
                list(self.multiPathInfoObject.combinations_routing_probabilities_dict.values())[0]):
            self.routingBlocksCount += 1
            num_of_routes = arr.shape[1]
            self.maxEntropies.append(-np.log(1.0 / num_of_routes))
            self.optimization_bounds_continuous["entropy_block_{0}".format(idx)] = (0.0, self.maxEntropies[idx])
        self.entropiesPerLevelSorted = []
        for block_id in range(self.routingBlocksCount):
            ents = []
            for list_of_entropies in self.multiPathInfoObject.combinations_routing_entropies_dict.values():
                ents.extend(list_of_entropies[block_id].tolist())
                # for arr in list_of_entropies[block_id]:
                #     ents.add(arr)
            ents = set(ents)
            self.entropiesPerLevelSorted.append(sorted(list(ents)))
        print("X")

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
        val_indices, test_indices = train_test_split(indices, train_size=val_sample_count)
        return total_sample_count, val_indices, test_indices

    def measure_model_accuracy(self, model, dataset):
        training_accuracy, training_info_gain_list = model.evaluate(
            x=dataset.trainDataTf, epoch_id=0, dataset_type="training")
        test_accuracy, test_info_gain_list = model.evaluate(
            x=dataset.testDataTf, epoch_id=0, dataset_type="test")
        print("training_accuracy={0}".format(training_accuracy))
        print("test_accuracy={0}".format(test_accuracy))

    def get_dataset(self):
        return object()

    def get_model(self, model_id):
        return object()

    def run(self):
        list_of_entropy_thresholds = [
            [np.array([0.1, self.maxEntropies[0]])], [np.array([0.05, self.maxEntropies[1]])]]

        list_of_probability_thresholds = [
            np.stack(
                [np.array([0.4, 0.2]), np.array([0.3, 0.1])], axis=0),
            np.stack([0.85 * np.array([0.2, 0.3, 0.35, 0.15]), 0.85 * np.array([0.25, 0.05, 0.4, 0.3])], axis=0)]

        print("Use Numpy")
        for _ in range(10):
            t0 = time.time()
            self.multiPathInfoObject.measure_performance(cigt=self.model,
                                                         list_of_probability_thresholds=list_of_probability_thresholds,
                                                         list_of_entropy_intervals=list_of_entropy_thresholds,
                                                         indices=np.arange(self.totalSampleCount),
                                                         use_numpy_approach=True)
            t1 = time.time()
            print("measure_performance time:{0}".format(t1 - t0))

        print("Use Dict lookup")
        for _ in range(10):
            t0 = time.time()
            self.multiPathInfoObject.measure_performance(cigt=self.model,
                                                         list_of_probability_thresholds=list_of_probability_thresholds,
                                                         list_of_entropy_intervals=list_of_entropy_thresholds,
                                                         indices=np.arange(self.totalSampleCount),
                                                         use_numpy_approach=False)
            t1 = time.time()
            print("measure_performance time:{0}".format(t1 - t0))


        # self.routingProbabilities, self.routingEntropies, self.logits, self.groundTruths, self. \
        #     fullTrainingAccuracy, self.fullTestAccuracy = self.get_model_outputs()
        # probabilities_arr = list(self.routingProbabilities.values())[0]
        # self.maxEntropies = []
        # self.optimization_bounds_continuous = {}
        # self.routingBlocksCount = 0
        # for idx, arr in enumerate(probabilities_arr):
        #     self.routingBlocksCount += 1
        #     num_of_routes = arr.shape[1]
        #     self.maxEntropies.append(-np.log(1.0 / num_of_routes))
        #     self.optimization_bounds_continuous["entropy_block_{0}".format(idx)] = (0.0, self.maxEntropies[idx])
        # self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()
        # self.listOfEntropiesPerLevel, self.fullEntropyArray = self.prepare_entropies_per_level_and_decision()
        # self.routingCorrectnessDict, self.routingMacDict, self.valBaseAccuracy, self.testBaseAccuracy, self. \
        #     fullAccuracyArray, self.fullMacArray = self.get_correctness_and_mac_dicts()
        # self.reformat_arrays()
        # self.runId = DbLogger.get_run_id()
        # kv_rows = [(self.runId, "Validation Sample Count", "{0}".format(len(self.valIndices))),
        #            (self.runId, "Test Sample Count", "{0}".format(len(self.testIndices))),
        #            (self.runId, "Validation Base Accuracy", "{0}".format(self.valBaseAccuracy)),
        #            (self.runId, "Test Base Accuracy", "{0}".format(self.testBaseAccuracy)),
        #            (self.runId, "Full Test Accuracy", "{0}".format(self.fullTestAccuracy)),
        #            (self.runId, "xi", "{0}".format(self.xi)),
        #            (self.runId, "init_points", "{0}".format(self.init_points)),
        #            (self.runId, "n_iter", "{0}".format(self.n_iter)),
        #            (self.runId, "accuracyMacBalanceCoeff", "{0}".format(self.accuracyMacBalanceCoeff)),
        #            (self.runId, "modelId", "{0}".format(self.modelId)),
        #            (self.runId, "Base Mac", "{0}".format(self.routingMacDict[(0, 0)]))
        #            ]
        # DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
