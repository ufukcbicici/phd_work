import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.algorithms.constant_distribution import ConstantDistribution
from tf_2_cign.cigt.algorithms.sigmoid_mixture_of_gaussians import SigmoidMixtureOfGaussians
from tf_2_cign.cigt.algorithms.sigmoid_normal_distribution import SigmoidNormalDistribution
from tf_2_cign.cigt.cross_entropy_optimizers.cross_entropy_threshold_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities


class HighEntropyThresholdOptimizer(CrossEntropySearchOptimizer):
    def __init__(self, num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                 entropy_threshold_percentiles,
                 entropy_threshold_counts_after_percentiles,
                 are_entropy_thresholds_fixed, image_output_path, random_seed,
                 num_of_gmm_components_per_block):
        self.numOfGmmComponentsPerBlock = num_of_gmm_components_per_block
        self.entropyThresholdPercentiles = entropy_threshold_percentiles
        entropy_threshold_counts = [cnt + 1 for cnt in entropy_threshold_counts_after_percentiles]
        super().__init__(num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                         entropy_threshold_counts, are_entropy_thresholds_fixed, image_output_path, random_seed)

    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation += super().get_explanation_string()
        explanation = self.add_explanation(name_of_param="Search Method",
                                           value="HighEntropyThresholdOptimizer",
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="numOfGmmComponentsPerBlock",
                                           value=self.numOfGmmComponentsPerBlock,
                                           explanation=explanation, kv_rows=kv_rows)
        return explanation

    def init_probability_distributions(self):
        assert len(self.entropyThresholdCounts) == len(self.pathCounts) - 1
        for block_id in range(len(self.pathCounts) - 1):
            n_gmm_components = self.numOfGmmComponentsPerBlock[block_id]
            block_entropy_distributions = []
            curr_lower_bound = 0.0
            hard_threshold_rank = int(
                        len(self.listOfEntropiesSorted[block_id]) *
                        self.entropyThresholdPercentiles[block_id])
            threshold_step_count = (len(self.listOfEntropiesSorted[block_id]) -
                                    hard_threshold_rank) // self.entropyThresholdCounts[block_id]

            for threshold_id in range(self.entropyThresholdCounts[block_id]):
                if threshold_id == 0:
                    entropy_threshold = self.listOfEntropiesSorted[block_id][hard_threshold_rank]
                    distribution = ConstantDistribution(value=entropy_threshold)
                    curr_lower_bound = entropy_threshold
                else:
                    if threshold_id < self.entropyThresholdCounts[block_id] - 1:
                        higher_bound = self.listOfEntropiesSorted[block_id][hard_threshold_rank +
                                                                            threshold_id*threshold_step_count]
                    else:
                        higher_bound = self.maxEntropies[block_id]
                    distribution = SigmoidMixtureOfGaussians(num_of_components=n_gmm_components,
                                                             low_end=curr_lower_bound,
                                                             high_end=higher_bound)
                    curr_lower_bound = higher_bound
                block_entropy_distributions.append(distribution)
            self.entropyThresholdDistributions.append(block_entropy_distributions)
            # if not self.areEntropyThresholdsFixed:
            #     entropy_chunks = Utilities.divide_array_into_chunks(arr=self.listOfEntropiesSorted[block_id],
            #                                                         count=self.entropyThresholdCounts[block_id])
            #     # Entropy distributions
            #     lower_bound = 0.0
            #     higher_bound = self.maxEntropies[block_id]
            #     sigmoid_normal_dist = SigmoidNormalDistribution(low_end=lower_bound, high_end=higher_bound)
            #     block_entropy_distributions.append(sigmoid_normal_dist)
            # else:
            #     # Interpret entropyThresholdCounts as percentiles
            #     entropy_threshold = self.entropyThresholdCounts[block_id] * self.maxEntropies[block_id]
            #     distribution = ConstantDistribution(value=entropy_threshold)
            #     block_entropy_distributions.append(distribution)
            # self.entropyThresholdDistributions.append(block_entropy_distributions)

            # # Probability Threshold Distributions
            block_probability_threshold_distributions = []
            for probability_interval_id in range(self.entropyThresholdCounts[block_id] + 1):
                interval_distributions = []
                if probability_interval_id == 0:
                    for route_id in range(self.pathCounts[block_id + 1]):
                        distribution = ConstantDistribution(value=2.0)
                        interval_distributions.append(distribution)
                else:
                    # A separate threshold for every route
                    for route_id in range(self.pathCounts[block_id + 1]):
                        # distribution = ConstantDistribution(value=2.0)
                        # interval_distributions.append(distribution)
                        distribution = SigmoidMixtureOfGaussians(num_of_components=n_gmm_components,
                                                                 low_end=0.0, high_end=1.0)
                        # distribution = SigmoidNormalDistribution(low_end=0.0, high_end=1.0)
                        interval_distributions.append(distribution)
                block_probability_threshold_distributions.append(interval_distributions)
            self.probabilityThresholdDistributions.append(block_probability_threshold_distributions)
