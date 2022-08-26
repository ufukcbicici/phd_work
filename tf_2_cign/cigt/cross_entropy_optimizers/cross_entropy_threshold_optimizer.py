import numpy as np
import tensorflow as tf
import os
import time
from tqdm import tqdm
from mpire import WorkerPool
from sklearn.model_selection import train_test_split

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities


class CrossEntropySearchOptimizer(object):
    intermediate_outputs_path = os.path.join(os.path.dirname(__file__), "..", "intermediate_outputs")

    def __init__(self, num_of_epochs, accuracy_mac_balance_coeff, model_loader,
                 model_id, val_ratio,
                 entropy_threshold_counts, image_output_path, random_seed, meta_data=""):
        self.runId = DbLogger.get_run_id()
        self.explanationString = self.get_explanation_string()
        DbLogger.write_into_table(rows=[(self.runId, self.explanationString)], table=DbLogger.runMetaData)
        self.numOfEpochs = num_of_epochs
        self.randomSeed = random_seed
        self.accuracyMacBalanceCoeff = accuracy_mac_balance_coeff
        self.modelLoader = model_loader
        self.modelId = model_id
        self.valRatio = val_ratio
        self.entropyThresholdCounts = entropy_threshold_counts
        self.imageOutputPath = image_output_path
        self.model, self.dataset = self.modelLoader.get_model(model_id=self.modelId)
        self.pathCounts = list(self.model.pathCounts)
        self.totalSampleCount, self.valIndices, self.testIndices = None, None, None
        self.listOfEntropiesSorted = []
        self.maxEntropies = []
        self.entropyThresholdDistributions = []
        self.probabilityThresholdDistributions = []
        self.multiPathInfoObject = self.load_multipath_info()
        self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()
        self.get_sorted_entropy_lists()
        self.init_probability_distributions()
        self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=np.arange(self.totalSampleCount))
        self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.valIndices)
        self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.testIndices)

    def add_explanation(self, name_of_param, value, explanation, kv_rows):
        explanation += "{0}:{1}\n".format(name_of_param, value)
        kv_rows.append((self.runId, name_of_param, "{0}".format(value)))
        return explanation

    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation = self.add_explanation(name_of_param="Model Id",
                                           value=self.modelId,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Run Id", value=self.runId,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Val Ratio", value=self.valRatio,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="entropy_threshold_counts",
                                           value=self.entropyThresholdCounts,
                                           explanation=explanation, kv_rows=kv_rows)
        return explanation

    # Load routing information for the particular model
    def load_multipath_info(self):
        object_folder_path = os.path.join(CrossEntropySearchOptimizer.intermediate_outputs_path,
                                          "{0}".format(self.modelId))
        object_path = os.path.join(object_folder_path, "multipath_info_object.pkl")
        if not os.path.isdir(object_folder_path):
            os.mkdir(object_folder_path)
            multipath_info_object = MultipathCombinationInfo(batch_size=self.model.batchSize,
                                                             path_counts=self.pathCounts)
            multipath_info_object.generate_routing_info(cigt=self.model,
                                                        dataset=self.dataset.testDataTf)
            Utilities.pickle_save_to_file(path=object_path, file_content=multipath_info_object)
        else:
            multipath_info_object = Utilities.pickle_load_from_file(path=object_path)
        multipath_info_object.assert_routing_validity(cigt=self.model)
        multipath_info_object.assess_accuracy()
        return multipath_info_object

    def get_sorted_entropy_lists(self):
        assert self.valIndices is not None
        for block_id in range(len(self.pathCounts) - 1):
            next_block_path_count = self.pathCounts[block_id + 1]
            max_entropy = -np.log(1.0 / next_block_path_count)
            ents = []
            for list_of_entropies in self.multiPathInfoObject.combinations_routing_entropies_dict.values():
                entropy_list = list_of_entropies[block_id][self.valIndices].tolist()
                ents.extend(entropy_list)
            ents.append(max_entropy)
            self.maxEntropies.append(max_entropy)
            ents = list(set(ents))
            entropies_sorted = sorted(ents)
            self.listOfEntropiesSorted.append(entropies_sorted)

    def init_probability_distributions(self):
        pass

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

    @staticmethod
    def sample_parameters(path_counts,
                          entropy_threshold_distributions,
                          max_entropies,
                          probability_threshold_distributions):
        routing_blocks_count = len(path_counts) - 1
        list_of_entropy_thresholds = []
        list_of_probability_thresholds = []
        for block_id in range(routing_blocks_count):
            # Sample entropy intervals
            entropy_interval_higher_ends = []
            for entropy_threshold_id in range(len(entropy_threshold_distributions[block_id])):
                entropy_threshold = \
                    entropy_threshold_distributions[block_id][entropy_threshold_id].sample(num_of_samples=1)[0]
                entropy_interval_higher_ends.append(entropy_threshold)
            entropy_interval_higher_ends.append(max_entropies[block_id])
            list_of_entropy_thresholds.append(np.array(entropy_interval_higher_ends))

            # Sample probability thresholds
            probability_thresholds_for_block = []
            for interval_id in range(len(entropy_threshold_distributions[block_id]) + 1):
                interval_threshold_distributions = probability_threshold_distributions[block_id][interval_id]
                interval_thresholds = []
                for path_id in range(len(interval_threshold_distributions)):
                    probability_threshold = interval_threshold_distributions[path_id].sample(num_of_samples=1)[0]
                    interval_thresholds.append(probability_threshold)
                interval_thresholds = np.array(interval_thresholds)
                probability_thresholds_for_block.append(interval_thresholds)
            probability_thresholds_for_block = np.stack(probability_thresholds_for_block, axis=0)
            list_of_probability_thresholds.append(probability_thresholds_for_block)
        return list_of_entropy_thresholds, list_of_probability_thresholds

    @staticmethod
    def measure_performance(multipath_routing_info_obj,
                            path_counts,
                            list_of_entropy_thresholds,
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
            entropy_intervals = list_of_entropy_thresholds[block_id]
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

    @staticmethod
    def sample_and_evaluate(shared_objects, sample_count):
        multipath_routing_info_obj = shared_objects[0]
        val_indices = shared_objects[1]
        test_indices = shared_objects[2]
        path_counts = shared_objects[3]
        entropy_threshold_distributions = shared_objects[4]
        max_entropies = shared_objects[5]
        probability_threshold_distributions = shared_objects[6]
        balance_coeff = shared_objects[7]
        samples_list = []
        for sample_id in tqdm(range(sample_count)):
            e, p = CrossEntropySearchOptimizer.sample_parameters(
                path_counts=path_counts,
                entropy_threshold_distributions=entropy_threshold_distributions,
                max_entropies=max_entropies,
                probability_threshold_distributions=probability_threshold_distributions
            )
            val_accuracy, val_mean_mac, val_score = CrossEntropySearchOptimizer.measure_performance(
                path_counts=path_counts,
                multipath_routing_info_obj=multipath_routing_info_obj,
                list_of_probability_thresholds=p,
                list_of_entropy_thresholds=e,
                indices=val_indices,
                use_numpy_approach=True,
                balance_coeff=balance_coeff)
            test_accuracy, test_mean_mac, test_score = CrossEntropySearchOptimizer.measure_performance(
                path_counts=path_counts,
                multipath_routing_info_obj=multipath_routing_info_obj,
                list_of_probability_thresholds=p,
                list_of_entropy_thresholds=e,
                indices=test_indices,
                use_numpy_approach=True,
                balance_coeff=balance_coeff)
            sample_dict = {
                "sample_id": sample_id,
                "entropy_intervals": e,
                "probability_thresholds": p,
                "val_accuracy": val_accuracy,
                "val_mean_mac": val_mean_mac,
                "val_score": val_score,
                "test_accuracy": test_accuracy,
                "test_mean_mac": test_mean_mac,
                "test_score": test_score
            }
            samples_list.append(sample_dict)
        return samples_list

    def run(self):
        epoch_count = 1000
        sample_count = 100000
        smoothing_coeff = 0.85
        gamma = 0.01
        n_jobs = 8
        sample_counts = [int(sample_count / n_jobs) for _ in range(n_jobs)]
        shared_objects = (self.multiPathInfoObject,
                          self.valIndices,
                          self.testIndices,
                          self.pathCounts,
                          self.entropyThresholdDistributions,
                          self.maxEntropies,
                          self.probabilityThresholdDistributions,
                          self.accuracyMacBalanceCoeff)

        percentile_count = int(gamma * sample_count)

        for epoch_id in range(epoch_count):
            # Single Thread
            if n_jobs == 1:
                samples_list = CrossEntropySearchOptimizer.sample_and_evaluate(
                    shared_objects=shared_objects, sample_count=100000
                )
            else:
                with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects) as pool:
                    results = pool.map(CrossEntropySearchOptimizer.sample_and_evaluate,
                                       sample_counts, progress_bar=True)
                print(results.__class__)
                print(len(results))
                samples_list = []
                for res_arr in results:
                    samples_list.extend(res_arr)

            samples_sorted = sorted(samples_list, key=lambda d_: d_["val_score"], reverse=True)
            val_accuracies = [d_["val_accuracy"] for d_ in samples_sorted]
            test_accuracies = [d_["test_accuracy"] for d_ in samples_sorted]
            val_test_corr = np.corrcoef(val_accuracies, test_accuracies)[0, 1]
            mean_val_acc = np.mean(val_accuracies)
            mean_test_acc = np.mean(test_accuracies)
            mean_val_mac = np.mean([d_["val_mean_mac"] for d_ in samples_sorted])
            mean_test_mac = np.mean([d_["test_mean_mac"] for d_ in samples_sorted])

            print("Epoch:{0} val_test_corr={1}".format(epoch_id, val_test_corr))
            print("Epoch:{0} mean_val_acc={1}".format(epoch_id, mean_val_acc))
            print("Epoch:{0} mean_test_acc={1}".format(epoch_id, mean_test_acc))
            print("Epoch:{0} mean_val_mac={1}".format(epoch_id, mean_val_mac))
            print("Epoch:{0} mean_test_mac={1}".format(epoch_id, mean_test_mac))

            samples_gamma = samples_sorted[0:percentile_count]
            val_accuracies_gamma = [d_["val_accuracy"] for d_ in samples_gamma]
            test_accuracies_gamma = [d_["test_accuracy"] for d_ in samples_gamma]
            val_test_gamma_corr = np.corrcoef(val_accuracies_gamma, test_accuracies_gamma)[0, 1]
            mean_val_gamma_acc = np.mean(val_accuracies_gamma)
            mean_test_gamma_acc = np.mean(test_accuracies_gamma)
            mean_val_gamma_mac = np.mean([d_["val_mean_mac"] for d_ in samples_gamma])
            mean_test_gamma_mac = np.mean([d_["test_mean_mac"] for d_ in samples_gamma])

            print("Epoch:{0} val_test_gamma_corr={1}".format(epoch_id, val_test_gamma_corr))
            print("Epoch:{0} mean_val_gamma_acc={1}".format(epoch_id, mean_val_gamma_acc))
            print("Epoch:{0} mean_test_gamma_acc={1}".format(epoch_id, mean_test_gamma_acc))
            print("Epoch:{0} mean_val_gamma_mac={1}".format(epoch_id, mean_val_gamma_mac))
            print("Epoch:{0} mean_test_gamma_mac={1}".format(epoch_id, mean_test_gamma_mac))

            DbLogger.write_into_table(rows=
                                      [(self.runId,
                                        epoch_id,
                                        np.asscalar(val_test_corr),
                                        np.asscalar(mean_val_acc),
                                        np.asscalar(mean_test_acc),
                                        np.asscalar(mean_val_mac),
                                        np.asscalar(mean_test_mac),
                                        np.asscalar(val_test_gamma_corr),
                                        np.asscalar(mean_val_gamma_acc),
                                        np.asscalar(mean_test_gamma_acc),
                                        np.asscalar(mean_val_gamma_mac),
                                        np.asscalar(mean_test_gamma_mac),
                                        self.modelId
                                        )], table="ce_logs_table")

            routing_blocks_count = len(self.pathCounts) - 1
            for block_id in range(routing_blocks_count):
                # Entropy distributions
                for entropy_threshold_id in range(len(self.entropyThresholdDistributions[block_id])):
                    print("ML Block:{0} Entropy:{1}".format(block_id, entropy_threshold_id))
                    data = []
                    for d_ in samples_gamma:
                        assert len(d_["entropy_intervals"][block_id]) \
                               == len(self.entropyThresholdDistributions[block_id]) + 1
                        data.append(d_["entropy_intervals"][block_id][entropy_threshold_id])
                    self.entropyThresholdDistributions[block_id][entropy_threshold_id].maximum_likelihood_estimate(
                        data=np.array(data), alpha=smoothing_coeff)
                # print("X")
                # Probability distributions
                probability_thresholds_for_block = []
                for interval_id in range(len(self.entropyThresholdDistributions[block_id]) + 1):
                    interval_threshold_distributions = self.probabilityThresholdDistributions[block_id][interval_id]
                    for path_id in range(len(interval_threshold_distributions)):
                        print("ML Block:{0} Interval:{1} Path:{2}".format(block_id, interval_id, path_id))
                        data = []
                        for d_ in samples_gamma:
                            data.append(d_["probability_thresholds"][block_id][interval_id, path_id])
                        self.probabilityThresholdDistributions[block_id][
                            interval_id][path_id].maximum_likelihood_estimate(data=np.array(data),
                                                                              alpha=smoothing_coeff)
        print("X")

    # CREATE
    # TABLE
    # "ce_logs_table"(
    #     "run_id"
    # INTEGER,
    # "epoch"
    # INTEGER,
    # "val_test_corr"
    # NUMERIC,
    # "mean_val_acc"
    # NUMERIC,
    # "mean_test_acc"
    # NUMERIC,
    # "mean_val_mac"
    # NUMERIC,
    # "mean_test_mac"
    # NUMERIC,
    # "val_test_gamma_corr"
    # NUMERIC,
    # "mean_val_gamma_acc"
    # NUMERIC,
    # "mean_test_gamma_acc"
    # NUMERIC,
    # "mean_val_gamma_mac"
    # NUMERIC,
    # "mean_test_gamma_mac"
    # NUMERIC
    # )