import tensorflow as tf
import numpy as np
import time

from algorithms.dataset_linking_algorithm import DatasetLinkingAlgorithm
from algorithms.threshold_optimization_algorithms.deep_q_networks.deep_q_threshold_optimizer import \
    DeepQThresholdOptimizer
from algorithms.threshold_optimization_algorithms.deep_q_networks.multi_iteration_deep_q_learning import \
    MultiIterationDQN
from algorithms.threshold_optimization_algorithms.deep_q_networks.q_learning_threshold_optimizer import \
    QLearningThresholdOptimizer
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

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
    network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    routing_data = DatasetLinkingAlgorithm.link_dataset_v3(network_name_="FashionNet_Lite", run_id_=453,
                                                           degree_list_=[2, 2],
                                                           test_iterations_=[48000])
    routing_data.apply_validation_test_split(test_ratio=0.1)
    dqn = MultiIterationDQN(routing_dataset=routing_data, network=network, network_name=network_name,
                            run_id=453, used_feature_names=used_output_names)
    print("X")
    # q_learning_threshold_optimizer = DeepQThresholdOptimizer(
    #     validation_data=validation_data,
    #     test_data=test_data, network=network, network_name=network_name, run_id=network_id, lambda_mac_cost=0.5,
    #     q_learning_func="cnn", used_feature_names=used_output_names)
    # q_learning_threshold_optimizer.train(level=1,
    #                                      sample_count=512,
    #                                      episode_count=25000,
    #                                      discount_factor=1.0,
    #                                      epsilon_discount_factor=0.99975,
    #                                      learning_rate=0.001
    #                                      )
    #


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    train_deep_q_learning()


if __name__ == "__main__":
    main()
