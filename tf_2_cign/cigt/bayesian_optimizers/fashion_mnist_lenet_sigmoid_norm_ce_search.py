import os
import numpy as np
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
        initial_path = os.path.join(self.rootPath, "0")
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
                dist.plot_distribution(root_path=initial_path)
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
                    dist.plot_distribution(root_path=initial_path)
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
            for interval_id in range(len(entropy_interval_distributions[block_id] + 1)):
                probability_thresholds_for_e_id = []
                for path_id in range(path_counts[block_id + 1]):
                    probability_threshold = \
                        probability_threshold_distributions[block_id][interval_id][path_id].sample(num_of_samples=1)[0]
                    probability_thresholds_for_e_id.append(probability_threshold)
                probability_thresholds_for_block.append(np.array(probability_thresholds_for_e_id))
            probability_thresholds_for_block = np.stack(probability_thresholds_for_block, axis=0)
            list_of_probability_thresholds.append(probability_thresholds_for_block)
        return list_of_entropy_thresholds, list_of_probability_thresholds

    def run(self):
        epoch_count = 1000
        sample_count = 100000
        smoothing_coeff = 0.85
        gamma = 0.01
        n_jobs = 5
        sample_counts = [int(sample_count / n_jobs) for _ in range(n_jobs)]

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
            with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects) as pool:
                results = pool.map(FashionMnistLenetCrossEntropySearch.sample_from_search_parameters,
                                   sample_counts, progress_bar=True)
            print(results.__class__)
            print(len(results))
            samples_list = []
            for res_arr in results:
                samples_list.extend(res_arr)

            # Single Thread
            # results = FashionMnistLenetCrossEntropySearch.sample_from_search_parameters(
            #     shared_objects=shared_objects, sample_count=100000
            # )

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
                        data=data, alpha=smoothing_coeff)
                # print("X")
                # Probability distributions
                for e_id in range(len(self.entropyIntervalDistributions[block_id]) + 1):
                    for path_id in range(self.pathCounts[block_id + 1]):
                        p_id = self.pathCounts[block_id + 1] * e_id + path_id
                        data = []
                        for d_ in samples_gamma:
                            # assert len(d_["probability_thresholds"][block_id]) \
                            #        == len(self.entropyIntervalDistributions[block_id]) + 1
                            data.append(d_["probability_thresholds"][block_id][e_id, path_id])
                        self.probabilityThresholdDistributions[block_id][p_id].maximum_likelihood_estimate(
                            data=data, alpha=smoothing_coeff)
            #     print("X")
            # print("X")