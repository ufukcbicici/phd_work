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


class SigmoidGmmCeThresholdOptimizer(CrossEntropySearchOptimizer):
    def __init__(self, num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                 entropy_threshold_counts, are_entropy_thresholds_fixed, image_output_path, random_seed, n_jobs,
                 apply_temperature_optimization_to_entropies, apply_temperature_optimization_to_routing_probabilities,
                 num_of_gmm_components_per_block):
        self.numOfGmmComponentsPerBlock = num_of_gmm_components_per_block
        super().__init__(num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                         entropy_threshold_counts, are_entropy_thresholds_fixed, image_output_path, random_seed, n_jobs,
                         apply_temperature_optimization_to_entropies,
                         apply_temperature_optimization_to_routing_probabilities)

    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation += super().get_explanation_string()
        explanation = self.add_explanation(name_of_param="Search Method",
                                           value="SigmoidGmmCeThresholdOptimizer",
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="numOfGmmComponentsPerBlock",
                                           value=self.numOfGmmComponentsPerBlock,
                                           explanation=explanation, kv_rows=kv_rows)
        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation

    def init_probability_distributions(self):
        assert len(self.entropyThresholdCounts) == len(self.pathCounts) - 1
        for block_id in range(len(self.pathCounts) - 1):
            n_gmm_components = self.numOfGmmComponentsPerBlock[block_id]
            n_entropy_thresholds = self.entropyThresholdCounts[block_id]
            block_entropy_distributions = []
            if not self.areEntropyThresholdsFixed:
                entropy_chunks = Utilities.divide_array_into_chunks(arr=self.listOfEntropiesSorted[block_id],
                                                                    count=self.entropyThresholdCounts[block_id])
                # Entropy distributions
                curr_lower_bound = 0.0
                for entropy_threshold_id in range(n_entropy_thresholds):
                    curr_upper_bound = entropy_chunks[entropy_threshold_id][-1]
                    if n_gmm_components > 1:
                        distribution = SigmoidMixtureOfGaussians(num_of_components=n_gmm_components,
                                                                 low_end=curr_lower_bound,
                                                                 high_end=curr_upper_bound)
                    else:
                        distribution = SigmoidNormalDistribution(low_end=curr_lower_bound,
                                                                 high_end=curr_upper_bound)
                    curr_lower_bound = curr_upper_bound
                    block_entropy_distributions.append(distribution)
            else:
                entropy_chunks = Utilities.divide_array_into_chunks(arr=self.listOfEntropiesSorted[block_id],
                                                                    count=self.entropyThresholdCounts[block_id] + 1)
                for entropy_threshold_id in range(n_entropy_thresholds):
                    distribution = ConstantDistribution(value=entropy_chunks[entropy_threshold_id][-1])
                    block_entropy_distributions.append(distribution)

            self.entropyThresholdDistributions.append(block_entropy_distributions)

            # # Probability Threshold Distributions
            block_probability_threshold_distributions = []
            for probability_interval_id in range(n_entropy_thresholds + 1):
                # A separate threshold for every route
                interval_distributions = []
                for route_id in range(self.pathCounts[block_id + 1]):
                    if n_gmm_components > 1:
                        distribution = SigmoidMixtureOfGaussians(num_of_components=n_gmm_components,
                                                                 low_end=0.0, high_end=1.0)
                    else:
                        distribution = SigmoidNormalDistribution(low_end=0.0, high_end=1.0)
                    interval_distributions.append(distribution)
                block_probability_threshold_distributions.append(interval_distributions)
            self.probabilityThresholdDistributions.append(block_probability_threshold_distributions)
