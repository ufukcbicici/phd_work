import os
import tensorflow as tf
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer
from tf_2_cign.cigt.bayesian_optimizers.cross_entropy_search_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.data_classes.categorical_distribution import CategoricalDistribution
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities


class FashionMnistLenetCrossEntropySearch(CrossEntropySearchOptimizer):
    def __init__(self, xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio,
                 entropy_interval_counts, entropy_bins_count, probability_bins_count):
        super().__init__(xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio)
        self.entropyIntervalCounts = entropy_interval_counts # [4, 5] -> For 4 for first block, 5 for second block
        self.entropyBinsCount = entropy_bins_count
        self.probabilityBinsCounts = probability_bins_count
        assert len(self.entropyIntervalCounts) == self.routingBlocksCount
        # Prepare categorical distributions for each entropy level
        self.entropyIntervalDistributions = []
        for block_id in range(self.routingBlocksCount):
            level_wise_entropy_intervals = []
            list_of_entropies = Utilities.divide_array_into_chunks(arr=self.entropiesPerLevelSorted[block_id],
                                                                   count=self.entropyIntervalCounts[block_id])
            for entropy_list in list_of_entropies:
                entropy_interval_chunks = Utilities.divide_array_into_chunks(arr=entropy_list,
                                                                             count=self.entropyBinsCount)
                interval_end_points = sorted([intr_[-1] for intr_ in entropy_interval_chunks])
                categorical_distribution = CategoricalDistribution(categories=interval_end_points)
                level_wise_entropy_intervals.append(categorical_distribution)
            self.entropyIntervalDistributions.append(level_wise_entropy_intervals)
        print("X")

            # curr_index = 0
            # category_count = len(self.entropiesPerLevelSorted[block_id]) // self.entropyIntervalCounts[block_id]
            # category_count += 1
            # for id_in_block in range(self.entropyIntervalCounts[block_id]):
            #     start_index = curr_index
            #     end_index = start_index + category_count






            # for id_in_block in range(self.entropyIntervalCounts[block_id]):
            #     self.entropiesPerLevelSorted
            #     # CategoricalDistribution

        #     ents = []
        #     for list_of_entropies in self.multiPathInfoObject.combinations_routing_entropies_dict.values():
        #         ents.extend(list_of_entropies[block_id].tolist())
        #         # for arr in list_of_entropies[block_id]:
        #         #     ents.add(arr)
        #     ents = set(ents)
        #     self.entropiesPerLevelSorted.append(sorted(list(ents)))
        # print("X")


    def get_dataset(self):
        fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
        return fashion_mnist

    def get_model(self, model_id):
        X = 0.15
        Y = 3.7233209862205525  # kwargs["information_gain_balance_coefficient"]
        Z = 0.7564802988471849  # kwargs["decision_loss_coefficient"]
        W = 0.01

        FashionNetConstants.softmax_decay_initial = 25.0
        FashionNetConstants.softmax_decay_coefficient = 0.9999
        FashionNetConstants.softmax_decay_period = 1
        FashionNetConstants.softmax_decay_min_limit = 0.1
        softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=FashionNetConstants.softmax_decay_initial,
            decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
            decay_period=FashionNetConstants.softmax_decay_period,
            decay_min_limit=FashionNetConstants.softmax_decay_min_limit)

        learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                     value=W,
                                                     schedule=[(15000 + 12000, (1.0 / 2.0) * W),
                                                               (30000 + 12000, (1.0 / 4.0) * W),
                                                               (40000 + 12000, (1.0 / 40.0) * W)])
        print(learning_rate_calculator)

        with tf.device("GPU"):
            run_id = DbLogger.get_run_id()
            fashion_cigt = LenetCigt(batch_size=125,
                                     input_dims=(28, 28, 1),
                                     filter_counts=[32, 64, 128],
                                     kernel_sizes=[5, 5, 1],
                                     hidden_layers=[512, 256],
                                     decision_drop_probability=0.0,
                                     classification_drop_probability=X,
                                     decision_wd=0.0,
                                     classification_wd=0.0,
                                     decision_dimensions=[128, 128],
                                     class_count=10,
                                     information_gain_balance_coeff=Y,
                                     softmax_decay_controller=softmax_decay_controller,
                                     learning_rate_schedule=learning_rate_calculator,
                                     decision_loss_coeff=Z,
                                     path_counts=[2, 4],
                                     bn_momentum=0.9,
                                     warm_up_period=25,
                                     routing_strategy_name="Probability_Thresholds",
                                     run_id=run_id,
                                     evaluation_period=10,
                                     measurement_start=25,
                                     use_straight_through=True,
                                     optimizer_type="SGD",
                                     decision_non_linearity="Softmax",
                                     save_model=True,
                                     model_definition="Multipath Capacity with {0}".format("Probability_Thresholds"))
            weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
                                               "weights_{0}".format(model_id))
            fashion_cigt.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
            fashion_cigt.isInWarmUp = False
            weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
                                               "weights_{0}".format(model_id))
            fashion_cigt.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
            fashion_cigt.isInWarmUp = False
            return fashion_cigt
