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
                 entropy_interval_counts, entropy_bins_count, probability_bins_count):
        super().__init__(xi, init_points, n_iter, accuracy_mac_balance_coeff, model_id, val_ratio,
                         entropy_interval_counts, entropy_bins_count, probability_bins_count)
        self.entropyIntervalCounts = entropy_interval_counts  # [4, 5] -> For 4 for first block, 5 for second block
        self.entropyBinsCount = entropy_bins_count
        self.probabilityBinsCounts = probability_bins_count
        assert len(self.entropyIntervalCounts) == self.routingBlocksCount
        self.entropyIntervalDistributions = []
        self.probabilityThresholdDistributions = []
        self.create_probability_distributions()

    def create_probability_distributions(self):
        for block_id in range(self.routingBlocksCount):
            level_wise_entropy_distributions = []
            list_of_entropies = Utilities.divide_array_into_chunks(
                arr=self.entropiesPerLevelSorted[block_id],
                count=self.entropyIntervalCounts[block_id])
            # Distributions for entropy levels
            for threshold_id, entropy_list in enumerate(list_of_entropies):
                low_end = entropy_list[0]
                high_end = entropy_list[-1]
                dist = SigmoidNormalDistribution(
                    name="Entropy_Block{0}_EntropyThreshold{1}".format(block_id, threshold_id),
                    low_end=low_end, high_end=high_end)
                level_wise_entropy_distributions.append(dist)
            # Distributions for probability threshold levels.
            level_wise_prob_threshold_distributions = []
            for interval_id in range(len(list_of_entropies) + 1):
                path_threshold_distributions = []
                for path_id in range(self.model.pathCounts[block_id + 1]):
                    dist = SigmoidNormalDistribution(
                        name="ProbabilityThreshold_Block{0}_Interval{1}_Path{2}".format(
                            block_id, interval_id, path_id), low_end=0.0, high_end=1.0)
                    path_threshold_distributions.append(dist)
                level_wise_prob_threshold_distributions.append(path_threshold_distributions)
        print("X")




                # entropy_interval_chunks = Utilities.divide_array_into_chunks(arr=entropy_list,
                #                                                              count=self.entropyBinsCount)

        # Prepare categorical distributions for each entropy level
        # self.entropyIntervalDistributions = []
        # self.probabilityThresholdDistributions = []
        # probability_thresholds = list([(1.0 / self.probabilityBinsCounts) * idx
        #                                for idx in range(self.probabilityBinsCounts)])
        # for block_id in range(self.routingBlocksCount):
        #     level_wise_entropy_distributions = []
        #     list_of_entropies = Utilities.divide_array_into_chunks(arr=self.entropiesPerLevelSorted[block_id],
        #                                                            count=self.entropyIntervalCounts[block_id])
        #     # Distributions for entropy levels
        #     for entropy_list in list_of_entropies:
        #         entropy_interval_chunks = Utilities.divide_array_into_chunks(arr=entropy_list,
        #                                                                      count=self.entropyBinsCount)
        #         interval_end_points = sorted([intr_[-1] for intr_ in entropy_interval_chunks])
        #         categorical_distribution = CategoricalDistribution(categories=interval_end_points)
        #         level_wise_entropy_distributions.append(categorical_distribution)
        #     # Distributions for probability threshold levels.
        #     level_wise_prob_threshold_distributions = []
        #     for prob_threshold_id in range(self.model.pathCounts[block_id + 1] * (len(list_of_entropies) + 1)):
        #         categorical_distribution = CategoricalDistribution(categories=probability_thresholds)
        #         level_wise_prob_threshold_distributions.append(categorical_distribution)
        #     self.entropyIntervalDistributions.append(level_wise_entropy_distributions)
        #     self.probabilityThresholdDistributions.append(level_wise_prob_threshold_distributions

#     self.entropyIntervalCounts = entropy_interval_counts  # [4, 5] -> For 4 for first block, 5 for second block
#     self.entropyBinsCount = entropy_bins_count
#     self.probabilityBinsCounts = probability_bins_count
#     assert len(self.entropyIntervalCounts) == self.routingBlocksCount
#     # Prepare categorical distributions for each entropy level
#     self.entropyIntervalDistributions = []
#     self.probabilityThresholdDistributions = []
#     probability_thresholds = list([(1.0 / self.probabilityBinsCounts) * idx
#                                    for idx in range(self.probabilityBinsCounts)])
#     for block_id in range(self.routingBlocksCount):
#         level_wise_entropy_distributions = []
#         list_of_entropies = Utilities.divide_array_into_chunks(arr=self.entropiesPerLevelSorted[block_id],
#                                                                count=self.entropyIntervalCounts[block_id])
#         # Distributions for entropy levels
#         for entropy_list in list_of_entropies:
#             entropy_interval_chunks = Utilities.divide_array_into_chunks(arr=entropy_list,
#                                                                          count=self.entropyBinsCount)
#             interval_end_points = sorted([intr_[-1] for intr_ in entropy_interval_chunks])
#             categorical_distribution = CategoricalDistribution(categories=interval_end_points)
#             level_wise_entropy_distributions.append(categorical_distribution)
#         # Distributions for probability threshold levels.
#         level_wise_prob_threshold_distributions = []
#         for prob_threshold_id in range(self.model.pathCounts[block_id + 1] * (len(list_of_entropies) + 1)):
#             categorical_distribution = CategoricalDistribution(categories=probability_thresholds)
#             level_wise_prob_threshold_distributions.append(categorical_distribution)
#         self.entropyIntervalDistributions.append(level_wise_entropy_distributions)
#         self.probabilityThresholdDistributions.append(level_wise_prob_threshold_distributions)
#
# def get_dataset(self):
#     fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
#     return fashion_mnist
#
# def get_model(self, model_id):
#     X = 0.15
#     Y = 3.7233209862205525  # kwargs["information_gain_balance_coefficient"]
#     Z = 0.7564802988471849  # kwargs["decision_loss_coefficient"]
#     W = 0.01
#
#     FashionNetConstants.softmax_decay_initial = 25.0
#     FashionNetConstants.softmax_decay_coefficient = 0.9999
#     FashionNetConstants.softmax_decay_period = 1
#     FashionNetConstants.softmax_decay_min_limit = 0.1
#     softmax_decay_controller = StepWiseDecayAlgorithm(
#         decay_name="Stepwise",
#         initial_value=FashionNetConstants.softmax_decay_initial,
#         decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
#         decay_period=FashionNetConstants.softmax_decay_period,
#         decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
#
#     learning_rate_calculator = DiscreteParameter(name="lr_calculator",
#                                                  value=W,
#                                                  schedule=[(15000 + 12000, (1.0 / 2.0) * W),
#                                                            (30000 + 12000, (1.0 / 4.0) * W),
#                                                            (40000 + 12000, (1.0 / 40.0) * W)])
#     print(learning_rate_calculator)
#
#     with tf.device("GPU"):
#         run_id = DbLogger.get_run_id()
#         fashion_cigt = LenetCigt(batch_size=125,
#                                  input_dims=(28, 28, 1),
#                                  filter_counts=[32, 64, 128],
#                                  kernel_sizes=[5, 5, 1],
#                                  hidden_layers=[512, 256],
#                                  decision_drop_probability=0.0,
#                                  classification_drop_probability=X,
#                                  decision_wd=0.0,
#                                  classification_wd=0.0,
#                                  decision_dimensions=[128, 128],
#                                  class_count=10,
#                                  information_gain_balance_coeff=Y,
#                                  softmax_decay_controller=softmax_decay_controller,
#                                  learning_rate_schedule=learning_rate_calculator,
#                                  decision_loss_coeff=Z,
#                                  path_counts=[2, 4],
#                                  bn_momentum=0.9,
#                                  warm_up_period=25,
#                                  routing_strategy_name="Probability_Thresholds",
#                                  run_id=run_id,
#                                  evaluation_period=10,
#                                  measurement_start=25,
#                                  use_straight_through=True,
#                                  optimizer_type="SGD",
#                                  decision_non_linearity="Softmax",
#                                  save_model=True,
#                                  model_definition="Multipath Capacity with {0}".format("Probability_Thresholds"))
#         weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
#                                            "weights_{0}".format(model_id))
#         fashion_cigt.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
#         fashion_cigt.isInWarmUp = False
#         weights_folder_path = os.path.join(os.path.dirname(__file__), "..", "saved_models",
#                                            "weights_{0}".format(model_id))
#         fashion_cigt.load_weights(filepath=os.path.join(weights_folder_path, "fully_trained_weights"))
#         fashion_cigt.isInWarmUp = False
#         return fashion_cigt
#
# def run(self):
#     epoch_count = 1000
#     sample_count = 100000
#     smoothing_coeff = 0.85
#     gamma = 0.01
#     n_jobs = 5
#     sample_counts = [int(sample_count / n_jobs) for _ in range(n_jobs)]
#
#     shared_objects = (self.multiPathInfoObject,
#                       self.valIndices,
#                       self.testIndices,
#                       self.pathCounts,
#                       self.entropyIntervalDistributions,
#                       self.maxEntropies,
#                       self.probabilityThresholdDistributions)
#
#     percentile_count = int(gamma * sample_count)
#
#     for epoch_id in range(epoch_count):
#         with WorkerPool(n_jobs=n_jobs, shared_objects=shared_objects) as pool:
#             results = pool.map(FashionMnistLenetCrossEntropySearch.sample_from_search_parameters,
#                                sample_counts, progress_bar=True)
#         print(results.__class__)
#         print(len(results))
#         samples_list = []
#         for res_arr in results:
#             samples_list.extend(res_arr)
#
#         # Single Thread
#         # results = FashionMnistLenetCrossEntropySearch.sample_from_search_parameters(
#         #     shared_objects=shared_objects, sample_count=100000
#         # )
#
#         samples_sorted = sorted(samples_list, key=lambda d_: d_["val_score"], reverse=True)
#         val_accuracies = [d_["val_accuracy"] for d_ in samples_sorted]
#         test_accuracies = [d_["test_accuracy"] for d_ in samples_sorted]
#         val_test_corr = np.corrcoef(val_accuracies, test_accuracies)[0, 1]
#         mean_val_acc = np.mean(val_accuracies)
#         mean_test_acc = np.mean(test_accuracies)
#         mean_val_mac = np.mean([d_["val_mean_mac"] for d_ in samples_sorted])
#         mean_test_mac = np.mean([d_["test_mean_mac"] for d_ in samples_sorted])
#
#         print("Epoch:{0} val_test_corr={1}".format(epoch_id, val_test_corr))
#         print("Epoch:{0} mean_val_acc={1}".format(epoch_id, mean_val_acc))
#         print("Epoch:{0} mean_test_acc={1}".format(epoch_id, mean_test_acc))
#         print("Epoch:{0} mean_val_mac={1}".format(epoch_id, mean_val_mac))
#         print("Epoch:{0} mean_test_mac={1}".format(epoch_id, mean_test_mac))
#
#         samples_gamma = samples_sorted[0:percentile_count]
#         val_accuracies_gamma = [d_["val_accuracy"] for d_ in samples_gamma]
#         test_accuracies_gamma = [d_["test_accuracy"] for d_ in samples_gamma]
#         val_test_gamma_corr = np.corrcoef(val_accuracies_gamma, test_accuracies_gamma)[0, 1]
#         mean_val_gamma_acc = np.mean(val_accuracies_gamma)
#         mean_test_gamma_acc = np.mean(test_accuracies_gamma)
#         mean_val_gamma_mac = np.mean([d_["val_mean_mac"] for d_ in samples_gamma])
#         mean_test_gamma_mac = np.mean([d_["test_mean_mac"] for d_ in samples_gamma])
#
#         print("Epoch:{0} val_test_gamma_corr={1}".format(epoch_id, val_test_gamma_corr))
#         print("Epoch:{0} mean_val_gamma_acc={1}".format(epoch_id, mean_val_gamma_acc))
#         print("Epoch:{0} mean_test_gamma_acc={1}".format(epoch_id, mean_test_gamma_acc))
#         print("Epoch:{0} mean_val_gamma_mac={1}".format(epoch_id, mean_val_gamma_mac))
#         print("Epoch:{0} mean_test_gamma_mac={1}".format(epoch_id, mean_test_gamma_mac))
#
#         # print("X")
#         # Maximum Likelihood estimates for categorical distributions
#         routing_blocks_count = len(self.pathCounts) - 1
#         for block_id in range(routing_blocks_count):
#             # Entropy distributions
#             for entropy_interval_id in range(len(self.entropyIntervalDistributions[block_id])):
#                 data = []
#                 for d_ in samples_gamma:
#                     assert len(d_["entropy_intervals"][block_id]) \
#                            == len(self.entropyIntervalDistributions[block_id]) + 1
#                     data.append(d_["entropy_intervals"][block_id][entropy_interval_id])
#                 self.entropyIntervalDistributions[block_id][entropy_interval_id].maximum_likelihood_estimate(
#                     data=data, alpha=smoothing_coeff)
#             # print("X")
#             # Probability distributions
#             for e_id in range(len(self.entropyIntervalDistributions[block_id]) + 1):
#                 for path_id in range(self.pathCounts[block_id + 1]):
#                     p_id = self.pathCounts[block_id + 1] * e_id + path_id
#                     data = []
#                     for d_ in samples_gamma:
#                         # assert len(d_["probability_thresholds"][block_id]) \
#                         #        == len(self.entropyIntervalDistributions[block_id]) + 1
#                         data.append(d_["probability_thresholds"][block_id][e_id, path_id])
#                     self.probabilityThresholdDistributions[block_id][p_id].maximum_likelihood_estimate(
#                         data=data, alpha=smoothing_coeff)
#         #     print("X")
#         # print("X")
