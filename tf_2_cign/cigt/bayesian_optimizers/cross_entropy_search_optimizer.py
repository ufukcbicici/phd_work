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
        self.pathCounts = list(self.model.pathCounts)
        self.measure_model_accuracy(model=self.model, dataset=self.dataset)
        self.multiPathInfoObject = MultipathCombinationInfo(batch_size=self.model.batchSize,
                                                            path_counts=self.pathCounts)
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
                entropy_list = list_of_entropies[block_id][self.valIndices].tolist()
                ents.extend(entropy_list)
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

    @staticmethod
    def sample_intervals(path_counts,
                         entropy_interval_distributions,
                         max_entropies,
                         probability_threshold_distributions):
        routing_blocks_count = len(path_counts) - 1
        list_of_entropy_thresholds = []
        list_of_probability_thresholds = []
        for block_id in range(routing_blocks_count):
            # Sample entropy intervals
            entropy_interval_higher_ends = []
            for entropy_interval_id in range(len(entropy_interval_distributions[block_id])):
                entropy_threshold = \
                    entropy_interval_distributions[block_id][entropy_interval_id].sample(num_of_samples=1)[0]
                entropy_interval_higher_ends.append(entropy_threshold)
            entropy_interval_higher_ends.append(max_entropies[block_id])
            list_of_entropy_thresholds.append(np.array(entropy_interval_higher_ends))

            # Sample probability thresholds
            block_list_for_probs = []
            for e_id in range(len(entropy_interval_higher_ends)):
                probability_thresholds_for_e_id = []
                for path_id in range(path_counts[block_id + 1]):
                    p_id = path_counts[block_id + 1] * e_id + path_id
                    probability_threshold = \
                        probability_threshold_distributions[block_id][p_id].sample(num_of_samples=1)[0]
                    probability_thresholds_for_e_id.append(probability_threshold)
                probability_thresholds_for_e_id = np.array(probability_thresholds_for_e_id)
                block_list_for_probs.append(probability_thresholds_for_e_id)
            block_list_for_probs = np.stack(block_list_for_probs, axis=0)
            list_of_probability_thresholds.append(block_list_for_probs)
        return list_of_entropy_thresholds, list_of_probability_thresholds

    @staticmethod
    def measure_performance(multipath_routing_info_obj,
                            path_counts,
                            list_of_entropy_intervals,
                            list_of_probability_thresholds,
                            indices,
                            balance_coeff,
                            use_numpy_approach=True):
        sample_paths = np.zeros(shape=(len(indices), 1), dtype=np.int32)
        sample_paths[:, 0] = indices
        past_num_of_routes = 0
        time_intervals = []

        for block_id, route_count in enumerate(path_counts[1:]):
            # Prepare all possible valid decisions that can be taken by samples in this stage of the CIGT, based on past
            # routing decisions.

            # 1) Get the entropy of each sample, based on the past decisions.
            t0 = time.time()
            index_arrays = tuple([sample_paths[:, idx] for idx in range(sample_paths.shape[1])])
            if not use_numpy_approach:
                curr_sample_entropies = np.apply_along_axis(
                    func1d=lambda row: multipath_routing_info_obj.past_decisions_entropies_dict[
                        tuple(row[:past_num_of_routes])][row[-1]], arr=sample_paths, axis=1)
            else:
                curr_sample_entropies = multipath_routing_info_obj.past_decisions_entropies_list[block_id][index_arrays]
            # 2) Find the relevant entropy intervals for each sample
            t1 = time.time()
            entropy_intervals = list_of_entropy_intervals[block_id]
            interval_selections = \
                np.less_equal(np.expand_dims(curr_sample_entropies, axis=-1), entropy_intervals)
            interval_ids = np.argmax(interval_selections, axis=1)
            # 3) Get the probability thresholds with respect to the selected entropy intervals.
            t2 = time.time()
            probability_thresholds = list_of_probability_thresholds[block_id]
            probability_thresholds_per_sample = probability_thresholds[interval_ids]
            # 4) Get the current routing probabilities of each sample, based on the paths taken so far.
            t3 = time.time()
            if not use_numpy_approach:
                curr_sample_routing_probabilities = np.apply_along_axis(
                    func1d=lambda row: multipath_routing_info_obj.past_decisions_routing_probabilities_dict[
                        tuple(row[:past_num_of_routes])][row[-1]], arr=sample_paths, axis=1)
            else:
                curr_sample_routing_probabilities = \
                    multipath_routing_info_obj.past_decisions_routing_probabilities_list[
                        block_id][index_arrays]
            assert curr_sample_routing_probabilities.shape == probability_thresholds_per_sample.shape
            # 5) Compare current routing probabilities to the thresholds
            t4 = time.time()
            route_selections = np.greater_equal(curr_sample_routing_probabilities,
                                                probability_thresholds_per_sample).astype(sample_paths.dtype)
            # 6) Get the maximum likely routing paths from the current routing probabilities
            argmax_indices = np.argmax(curr_sample_routing_probabilities, axis=1)
            ml_route_selections = np.zeros_like(route_selections)
            ml_route_selections[np.arange(ml_route_selections.shape[0], ), argmax_indices] = 1
            # 7) Detect degenerate cases where all routing probabilities are under routing thresholds and no path is
            # valid to route into. We are going to use the ML routing in such cases instead.
            num_of_selected_routes = np.sum(route_selections, axis=1)
            final_selections = np.where(np.expand_dims(num_of_selected_routes > 0, axis=-1),
                                        route_selections, ml_route_selections)
            route_selections = final_selections
            # 8) Integrate into the selected paths so far.
            sample_paths = np.concatenate([sample_paths[:, :-1], route_selections,
                                           np.expand_dims(indices, axis=1)], axis=1)
            past_num_of_routes += route_count

            # print("Block:{0} t1-t0:{1}".format(block_id, t1 - t0))
            # print("Block:{0} t2-t1:{1}".format(block_id, t2 - t1))
            # print("Block:{0} t3-t2:{1}".format(block_id, t3 - t2))
            # print("Block:{0} t4-t3:{1}".format(block_id, t4 - t3))

        # Calculate accuracy and mac cost
        base_mac = np.nanmin(multipath_routing_info_obj.past_decisions_mac_array)
        if not use_numpy_approach:
            validity_vector = np.apply_along_axis(
                func1d=lambda row: multipath_routing_info_obj.combinations_y_dict[
                                       tuple(row[:past_num_of_routes])][row[-1]]
                                   == np.argmax(multipath_routing_info_obj.combinations_y_hat_dict[
                                                    tuple(row[:past_num_of_routes])][row[-1]]),
                arr=sample_paths, axis=1)
            mac_vector = np.apply_along_axis(
                func1d=lambda row: multipath_routing_info_obj.past_decisions_mac_array[tuple(row[:past_num_of_routes])],
                arr=sample_paths, axis=1)
        else:
            index_arrays = tuple([sample_paths[:, idx] for idx in range(sample_paths.shape[1])])
            validity_vector = multipath_routing_info_obj.past_decisions_validity_array[index_arrays]
            mac_vector = multipath_routing_info_obj.past_decisions_mac_array[index_arrays[:-1]]

        dif_vector = mac_vector * (1.0 / base_mac)
        dif_vector = dif_vector - 1.0

        score_vector = balance_coeff * validity_vector - (1.0 - balance_coeff) * dif_vector

        accuracy = np.mean(validity_vector)
        mean_mac = np.mean(dif_vector)
        score = np.mean(score_vector)
        # print("accuracy={0} mean_mac={1} score={2}".format(accuracy, mean_mac, score))
        return accuracy, mean_mac, score


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
                                                         use_numpy_approach=True,
                                                         balance_coeff=1.0)
            t1 = time.time()
            print("measure_performance time:{0}".format(t1 - t0))

        print("Use Dict lookup")
        for _ in range(10):
            t0 = time.time()
            self.multiPathInfoObject.measure_performance(cigt=self.model,
                                                         list_of_probability_thresholds=list_of_probability_thresholds,
                                                         list_of_entropy_intervals=list_of_entropy_thresholds,
                                                         indices=np.arange(self.totalSampleCount),
                                                         use_numpy_approach=False,
                                                         balance_coeff=1.0)
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
