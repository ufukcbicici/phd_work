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
