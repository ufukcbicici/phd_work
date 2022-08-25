import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from mpire import WorkerPool

from tf_2_cign.cigt.algorithms.sigmoid_normal_distribution import SigmoidNormalDistribution
from tf_2_cign.cigt.bayesian_optimizers.cross_entropy_search_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.algorithms.categorical_distribution import CategoricalDistribution
from tf_2_cign.cigt.bayesian_optimizers.fashion_mnist_lenet_cross_entropy_search import \
    FashionMnistLenetCrossEntropySearch
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities
import time


class FashionMnistLenetSigmoidNormCeSearh(FashionMnistLenetCrossEntropySearch):
    def __init__(self, xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio,
                 entropy_interval_counts, entropy_bins_count, probability_bins_count, root_path):
        super().__init__(xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio,
                         entropy_interval_counts, entropy_bins_count, probability_bins_count)
        self.runId = DbLogger.get_run_id()
        DbLogger.write_into_table(rows=[(self.runId,
                                        "FashionMnistLenetSigmoidNormCeSearh")],
                                  table="run_meta_data")
        self.rootPath = os.path.join(root_path, str(self.runId))
        os.mkdir(self.rootPath)
        self.entropyIntervalCounts = entropy_interval_counts  # [4, 5] -> For 4 for first block, 5 for second block
        self.entropyBinsCount = entropy_bins_count
        self.probabilityBinsCounts = probability_bins_count
        assert len(self.entropyIntervalCounts) == self.routingBlocksCount
        self.entropyIntervalDistributions = []
        self.probabilityThresholdDistributions = []
        self.create_probability_distributions()

    def create_probability_distributions(self):
        initial_path = os.path.join(self.rootPath, "initial")
        os.mkdir(initial_path)
        for block_id in range(self.routingBlocksCount):
            level_wise_entropy_distributions = []
            list_of_entropies = Utilities.divide_array_into_chunks(
                arr=self.entropiesPerLevelSorted[block_id],
                count=self.entropyIntervalCounts[block_id])

            # Distributions for entropy levels
            for threshold_id in range(len(list_of_entropies)):
                if threshold_id == 0:
                    low_end = list_of_entropies[0][0]
                else:
                    low_end = level_wise_entropy_distributions[threshold_id-1].highEnd
                high_end = list_of_entropies[threshold_id][-1]

                dist = SigmoidNormalDistribution(
                    name="Entropy_Block{0}_EntropyThreshold{1}".format(block_id, threshold_id),
                    low_end=low_end, high_end=high_end)
                # dist.plot_distribution(root_path=initial_path)
                level_wise_entropy_distributions.append(dist)
            self.entropyIntervalDistributions.append(level_wise_entropy_distributions)

            # Distributions for probability threshold levels.
            level_wise_prob_threshold_distributions = []
            for interval_id in range(len(list_of_entropies) + 1):
                path_threshold_distributions = []
                for path_id in range(self.model.pathCounts[block_id + 1]):
                    dist = SigmoidNormalDistribution(
                        name="ProbabilityThreshold_Block{0}_Interval{1}_Path{2}".format(
                            block_id, interval_id, path_id), low_end=0.0, high_end=1.0)
                    # dist.plot_distribution(root_path=initial_path)
                    path_threshold_distributions.append(dist)
                level_wise_prob_threshold_distributions.append(path_threshold_distributions)
            self.probabilityThresholdDistributions.append(level_wise_prob_threshold_distributions)

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
            probability_thresholds_for_block = []
            for interval_id in range(len(entropy_interval_distributions[block_id]) + 1):
                probability_thresholds_for_e_id = []
                for path_id in range(path_counts[block_id + 1]):
                    probability_threshold = \
                        probability_threshold_distributions[block_id][interval_id][path_id].sample(num_of_samples=1)[0]
                    probability_thresholds_for_e_id.append(probability_threshold)
                probability_thresholds_for_block.append(np.array(probability_thresholds_for_e_id))
            probability_thresholds_for_block = np.stack(probability_thresholds_for_block, axis=0)
            list_of_probability_thresholds.append(probability_thresholds_for_block)
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

    @staticmethod
    def sample_from_search_parameters(shared_objects, sample_count):
        multipath_routing_info_obj = shared_objects[0]
        val_indices = shared_objects[1]
        test_indices = shared_objects[2]
        path_counts = shared_objects[3]
        entropy_interval_distributions = shared_objects[4]
        max_entropies = shared_objects[5]
        probability_threshold_distributions = shared_objects[6]
        samples_list = []
        for sample_id in tqdm(range(sample_count)):
            e, p = FashionMnistLenetSigmoidNormCeSearh.sample_intervals(
                path_counts=path_counts,
                entropy_interval_distributions=entropy_interval_distributions,
                max_entropies=max_entropies,
                probability_threshold_distributions=probability_threshold_distributions
            )
            val_accuracy, val_mean_mac, val_score = FashionMnistLenetSigmoidNormCeSearh.measure_performance(
                path_counts=path_counts,
                multipath_routing_info_obj=multipath_routing_info_obj,
                list_of_probability_thresholds=p,
                list_of_entropy_intervals=e,
                indices=val_indices,
                use_numpy_approach=True,
                balance_coeff=1.0)
            test_accuracy, test_mean_mac, test_score = FashionMnistLenetSigmoidNormCeSearh.measure_performance(
                path_counts=path_counts,
                multipath_routing_info_obj=multipath_routing_info_obj,
                list_of_probability_thresholds=p,
                list_of_entropy_intervals=e,
                indices=test_indices,
                use_numpy_approach=True,
                balance_coeff=1.0)
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
        self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.valIndices)
        self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.testIndices)

        e, p = self.sample_intervals(path_counts=self.pathCounts,
                                     entropy_interval_distributions=self.entropyIntervalDistributions,
                                     max_entropies=self.maxEntropies,
                                     probability_threshold_distributions=self.probabilityThresholdDistributions)
        shared_objects = (self.multiPathInfoObject,
                          self.valIndices,
                          self.testIndices,
                          self.pathCounts,
                          self.entropyIntervalDistributions,
                          self.maxEntropies,
                          self.probabilityThresholdDistributions)

        percentile_count = int(gamma * sample_count)

        for epoch_id in range(epoch_count):
            initial_path = os.path.join(self.rootPath, str(epoch_id))
            os.mkdir(initial_path)
            with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects) as pool:
                results = pool.map(FashionMnistLenetSigmoidNormCeSearh.sample_from_search_parameters,
                                   sample_counts, progress_bar=True)
            # Single Thread
            # results = FashionMnistLenetSigmoidNormCeSearh.sample_from_search_parameters(
            #     shared_objects=shared_objects, sample_count=100000
            # )
            # print(results.__class__)
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

            # print("X")
            # Maximum Likelihood estimates for categorical distributions
            routing_blocks_count = len(self.pathCounts) - 1
            for block_id in range(routing_blocks_count):
                # Entropy distributions
                for entropy_interval_id in range(len(self.entropyIntervalDistributions[block_id])):
                    data = []
                    for d_ in samples_gamma:
                        assert len(d_["entropy_intervals"][block_id]) \
                               == len(self.entropyIntervalDistributions[block_id]) + 1
                        data.append(d_["entropy_intervals"][block_id][entropy_interval_id])
                    self.entropyIntervalDistributions[block_id][entropy_interval_id].maximum_likelihood_estimate(
                        data=np.array(data), alpha=smoothing_coeff)
                    # self.entropyIntervalDistributions[block_id][entropy_interval_id].plot_distribution(
                    #     root_path=initial_path)
                # print("X")
                # Probability distributions
                for e_id in range(len(self.entropyIntervalDistributions[block_id]) + 1):
                    for path_id in range(self.pathCounts[block_id + 1]):
                        data = []
                        for d_ in samples_gamma:
                            # assert len(d_["probability_thresholds"][block_id]) \
                            #        == len(self.entropyIntervalDistributions[block_id]) + 1
                            data.append(d_["probability_thresholds"][block_id][e_id, path_id])
                        self.probabilityThresholdDistributions[block_id][e_id][path_id].maximum_likelihood_estimate(
                            data=np.array(data), alpha=smoothing_coeff)
                        # self.probabilityThresholdDistributions[block_id][e_id][path_id].plot_distribution(
                        #     root_path=initial_path)
            #     print("X")
            # print("X")