import tensorflow as tf
import numpy as np
import time

from algorithms.dataset_linking_algorithm import DatasetLinkingAlgorithm
from algorithms.threshold_optimization_algorithms.deep_q_networks.deep_q_threshold_optimizer import \
    DeepQThresholdOptimizer
from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
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
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


def train_basic_q_learning():
    network_id = 452
    network_name = "FashionNet_Lite"
    iteration = 47520

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
    network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    routing_data = network.load_routing_info(run_id=network_id, iteration=iteration, data_type="test",
                                             output_names=output_names)
    validation_data, test_data = routing_data.apply_validation_test_split(test_ratio=0.1)
    q_learning_threshold_optimizer = QLearningThresholdOptimizer(
        validation_data=validation_data,
        test_data=test_data, network=network, network_name=network_name, run_id=network_id, lambda_mac_cost=0.1,
        q_learning_func="cnn", used_feature_names=used_output_names)
    q_learning_threshold_optimizer.train(level=1,
                                         episode_count=1000000,
                                         discount_factor=1.0,
                                         epsilon_discount_factor=0.9999,
                                         learning_rate=0.001)
    print("X")


def train_deep_q_learning():
    network_id = 453
    network_name = "FashionNet_Lite"
    iteration = 47520
    list_of_l2_coeffs = [0.0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
                         0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005, 0.01]
    list_of_seeds = [67, 112, 42, 594713, 87, 1111, 484, 8779, 32999, 55123]
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
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
        routing_data.apply_validation_test_split(test_ratio=0.1)
        routing_data.switch_to_single_iteration_mode()
        dqn = DqnWithClassification(routing_dataset=routing_data, network=network, network_name=network_name,
                                    run_id=453, used_feature_names=used_output_names,
                                    dqn_func=DeepQNetworks.get_squeeze_and_excitation_block,
                                    lambda_mac_cost=0.0,
                                    valid_prediction_reward=1.0,
                                    invalid_prediction_penalty=0.0, feature_type="sum")
        # dqn = DqnWithReducedRegression(routing_dataset=routing_data, network=network, network_name=network_name,
        #                                run_id=453, used_feature_names=used_output_names,
        #                                dqn_func=DeepQNetworks.get_squeeze_and_excitation_block,
        #                                lambda_mac_cost=0.0,
        #                                valid_prediction_reward=1.0,
        #                                invalid_prediction_penalty=0.0, feature_type="sum")
        # dqn = DqnWithRegression(routing_dataset=routing_data, network=network, network_name=network_name,
        #                         run_id=453, used_feature_names=used_output_names, q_learning_func="cnn",
        #                         lambda_mac_cost=0.0,
        #                         valid_prediction_reward=1.0,
        #                         invalid_prediction_penalty=0.0,
        #                         invalid_action_penalty=-1.0,
        #                         feature_type="sum")
        # dqn = MultiIterationDQNRegression(routing_dataset=routing_data, network=network, network_name=network_name,
        #                                   run_id=453, used_feature_names=used_output_names, q_learning_func="cnn",
        #                                   lambda_mac_cost=0.2)
        dqn.train(level=1, sample_count=128, episode_count=50000, discount_factor=1.0, l2_lambda=l2_lambda, seed=seed)
        print("X")
        tf.reset_default_graph()
        break


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    train_deep_q_learning()


if __name__ == "__main__":
    main()
