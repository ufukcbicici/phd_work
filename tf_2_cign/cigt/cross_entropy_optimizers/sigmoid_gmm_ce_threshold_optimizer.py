import numpy as np
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.cross_entropy_optimizers.cross_entropy_threshold_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants


class SigmoidGmmCeThresholdOptimizer(CrossEntropySearchOptimizer):
    def __init__(self, num_of_epochs, accuracy_mac_balance_coeff, model_loader, model_id, val_ratio,
                 entropy_interval_counts, image_output_path):
        super().__init__(num_of_epochs, accuracy_mac_balance_coeff, model_loader, model_id, val_ratio,
                         entropy_interval_counts, image_output_path)

    def init_probability_distributions(self):
        assert len(self.entropyIntervalCounts) == len(self.pathCounts) - 1
        for block_id in range(len(self.pathCounts) - 1):

            # for list_of_entropies in self.multiPathInfoObject.combinations_routing_entropies_dict.values():
            #     entropy_list = list_of_entropies[block_id][self.valIndices].tolist()
            #     ents.extend(entropy_list)
            #     # for arr in list_of_entropies[block_id]:
            #     #     ents.add(arr)
            # ents = set(ents)
            # self.entropiesPerLevelSorted.append(sorted(list(ents)))

            # Entropy distributions
            num_of_entropy_intervals = self.entropyIntervalCounts[block_id]
            for entropy_interval_id in range(num_of_entropy_intervals):
                pass
