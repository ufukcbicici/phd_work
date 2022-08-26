import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.algorithms.sigmoid_mixture_of_gaussians import SigmoidMixtureOfGaussians
from tf_2_cign.cigt.cross_entropy_optimizers.cross_entropy_threshold_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities


class SigmoidGmmCeThresholdOptimizer(CrossEntropySearchOptimizer):
    def __init__(self, num_of_epochs, accuracy_mac_balance_coeff, model_loader, model_id, val_ratio,
                 entropy_threshold_counts, num_of_gmm_components_per_block, image_output_path, random_seed):
        self.numOfGmmComponentsPerBlock = num_of_gmm_components_per_block
        super().__init__(num_of_epochs, accuracy_mac_balance_coeff, model_loader, model_id, val_ratio,
                         entropy_threshold_counts, image_output_path, random_seed)

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
        return explanation

    def init_probability_distributions(self):
        assert len(self.entropyThresholdCounts) == len(self.pathCounts) - 1
        for block_id in range(len(self.pathCounts) - 1):
            entropy_chunks = Utilities.divide_array_into_chunks(arr=self.listOfEntropiesSorted[block_id],
                                                                count=self.entropyThresholdCounts[block_id])
            n_gmm_components = self.numOfGmmComponentsPerBlock[block_id]

            # Entropy distributions
            n_entropy_thresholds = self.entropyThresholdCounts[block_id]
            block_entropy_distributions = []
            curr_lower_bound = 0.0
            for entropy_threshold_id in range(n_entropy_thresholds):
                curr_upper_bound = entropy_chunks[entropy_threshold_id][-1]
                distribution = SigmoidMixtureOfGaussians(num_of_components=n_gmm_components,
                                                         low_end=curr_lower_bound,
                                                         high_end=curr_upper_bound)
                curr_lower_bound = curr_upper_bound
                block_entropy_distributions.append(distribution)
            self.entropyThresholdDistributions.append(block_entropy_distributions)

            # Probability Threshold Distributions
            block_probability_threshold_distributions = []
            for probability_interval_id in range(n_entropy_thresholds + 1):
                # A separate threshold for every route
                interval_distributions = []
                for route_id in range(self.pathCounts[block_id + 1]):
                    distribution = SigmoidMixtureOfGaussians(num_of_components=n_gmm_components,
                                                             low_end=0.0, high_end=1.0)
                    interval_distributions.append(distribution)
                block_probability_threshold_distributions.append(interval_distributions)
            self.probabilityThresholdDistributions.append(block_probability_threshold_distributions)


