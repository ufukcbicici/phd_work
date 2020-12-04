import tensorflow as tf
import numpy as np
import time

from algorithms.dataset_linking_algorithm import DatasetLinkingAlgorithm
from algorithms.threshold_optimization_algorithms.deep_q_networks.deep_q_threshold_optimizer import \
    DeepQThresholdOptimizer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_multi_level_classification import \
    DqnMultiLevelWithClassification
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_multi_level_reduced_regression import \
    DqnMultiLevelReducedRegression
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_classification import \
    DqnWithClassification
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_reduced_regression import \
    DqnWithReducedRegression
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_optimizer_with_regression import DqnWithRegression
from algorithms.threshold_optimization_algorithms.deep_q_networks.multi_iteration_deep_q_learning import \
    MultiIterationDQN
from algorithms.threshold_optimization_algorithms.deep_q_networks.multi_iteration_dqn_with_regression import \
    MultiIterationDQNRegression
from algorithms.threshold_optimization_algorithms.deep_q_networks.q_learning_threshold_optimizer import \
    QLearningThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.bayesian_optimization_with_clusters import \
    BayesianOptimizationWithClusters
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.ig_clustered_bayesian_optimization import \
    IgBasedBayesianOptimization
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.mixed_bayesian_optimizer import \
    MixedBayesianOptimizer
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


def train_direct_threshold_optimizer():
    network_id = 453
    network_name = "FashionNet_Lite"
    iteration = 47520
    list_of_l2_coeffs = [0.0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
                         0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005, 0.01]
    list_of_seeds = [67, 112, 42, 594713, 87, 1111, 484, 8779, 32999, 55123]
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]

    dqn_parameters = \
        {
            "LeNet_DQN":
                {
                    "CONV_FEATURES": [[32], [32]],
                    "HIDDEN_LAYERS": [[64, 32], [64, 32]],
                    "FILTER_SIZES": [[3], [3]],
                    "STRIDES": [[2], [2]],
                    "MAX_POOL": [[None], [None]]
                },
            "Squeeze_And_Excitation":
                {
                    "SE_REDUCTION_RATIO": [2, 2]
                }
        }

    param_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=[list_of_l2_coeffs, list_of_seeds])
    for param_tpl in param_tuples:
        l2_lambda = param_tpl[0]
        seed = param_tpl[1]
        np.random.seed(seed)
        network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
        routing_data = DatasetLinkingAlgorithm.link_dataset_v3(network_name_="FashionNet_Lite", run_id_=453,
                                                               degree_list_=[2, 2],
                                                               test_iterations_=[43680, 44160, 44640, 45120, 45600,
                                                                                 46080, 46560, 47040, 47520, 48000])
        routing_data.apply_validation_test_split(test_ratio=0.5)
        routing_data.switch_to_single_iteration_mode()

        # IgBasedBayesianOptimization.optimize(run_id=network_id, network=network,
        #                                      routing_data=routing_data, seed=seed, mixing_lambda=0.995)

        MixedBayesianOptimizer.optimize(optimization_iterations_count=3,
                                        run_id=network_id,
                                        network=network,
                                        iteration=0,
                                        routing_data=routing_data,
                                        seed=seed,
                                        cluster_count=5,
                                        fc_layers=[64, 32])

        # BayesianOptimizationWithClusters.optimize(mixing_lambda=1.0, iteration=0,
        #                                           cluster_count=3, fc_layers=[64, 32], run_id=network_id,
        #                                           network=network, routing_data=routing_data, seed=seed)

        # dto = DirectThresholdOptimizer(network=network, routing_data=routing_data, seed=seed)
        # dto.train(run_id=network_id, iteration=43680)
        # print("X")
        tf.reset_default_graph()
        break


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    train_direct_threshold_optimizer()


if __name__ == "__main__":
    main()
