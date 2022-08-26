import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.algorithms.categorical_distribution import CategoricalDistribution
from tf_2_cign.cigt.algorithms.sigmoid_mixture_of_gaussians import SigmoidMixtureOfGaussians
from tf_2_cign.cigt.cross_entropy_optimizers.cross_entropy_threshold_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities


class CategoricalCeThresholdOptimizer(CrossEntropySearchOptimizer):
    def __init__(self, num_of_epochs, accuracy_mac_balance_coeff, model_loader, model_id, val_ratio,
                 entropy_threshold_counts, image_output_path, random_seed, entropy_bins_count, probability_bins_count):
        self.entropyBinsCount = entropy_bins_count
        self.probabilityBinsCount = probability_bins_count
        super().__init__(num_of_epochs, accuracy_mac_balance_coeff, model_loader, model_id, val_ratio,
                         entropy_threshold_counts, image_output_path, random_seed)

    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation += super().get_explanation_string()
        explanation = self.add_explanation(name_of_param="Search Method",
                                           value="CategoricalCeThresholdOptimizer",
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="entropyBinsCount",
                                           value=self.entropyBinsCount,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="probabilityBinsCount",
                                           value=self.probabilityBinsCount,
                                           explanation=explanation, kv_rows=kv_rows)
        return explanation

    def init_probability_distributions(self):
        # Prepare categorical distributions for each entropy level
        self.entropyThresholdDistributions = []
        self.probabilityThresholdDistributions = []
        probability_thresholds = list([(1.0 / self.probabilityBinsCounts) * idx
                                       for idx in range(self.probabilityBinsCounts)])
        for block_id in range(len(self.pathCounts) - 1):
            list_of_lists_of_entropies = Utilities.divide_array_into_chunks(
                arr=self.listOfEntropiesSorted[block_id],
                count=self.entropyThresholdCounts[block_id])
            # Distributions for entropy levels
            level_wise_entropy_distributions = []
            n_entropy_thresholds = self.entropyThresholdCounts[block_id]
            assert n_entropy_thresholds == len(list_of_lists_of_entropies)
            for entropy_list in list_of_lists_of_entropies:
                entropy_interval_chunks = Utilities.divide_array_into_chunks(arr=entropy_list,
                                                                             count=self.entropyBinsCount)
                interval_end_points = sorted([intr_[-1] for intr_ in entropy_interval_chunks])
                categorical_distribution = CategoricalDistribution(categories=interval_end_points)
                level_wise_entropy_distributions.append(categorical_distribution)
            self.entropyThresholdDistributions.append(level_wise_entropy_distributions)
            # Distributions for probability threshold levels.
            level_wise_prob_threshold_distributions = []
            for probability_interval_id in range(n_entropy_thresholds + 1):
                # A separate threshold for every route
                interval_distributions = []
                for route_id in range(self.pathCounts[block_id + 1]):
                    distribution = CategoricalDistribution(categories=probability_thresholds)
                    interval_distributions.append(distribution)
                level_wise_prob_threshold_distributions.append(interval_distributions)
            self.probabilityThresholdDistributions.append(level_wise_prob_threshold_distributions)
