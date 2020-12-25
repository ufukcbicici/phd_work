import numpy as np
import tensorflow as tf

from algorithms.dataset_linking_algorithm import DatasetLinkingAlgorithm
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.kmeans_plus_bayesian_optimization import \
    KmeansPlusBayesianOptimization
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from auxillary.db_logger import DbLogger


def train_direct_threshold_optimizer():
    # Fashion MNIST
    # network_id = 453
    # network_name = "FashionNet_Lite"
    # iteration = 47520
    # list_of_l2_coeffs = [0.0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
    #                      0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005, 0.01]

    # USPS
    network_id = 1892
    network_name = "USPS_CIGN"
    list_of_l2_coeffs = [0.0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
                         0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005, 0.01]
    iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
                         11741, 11800])

    lambdas = [1.0, 0.99, 0.95, 0.9]
    xis = [0.0, 0.001, 0.01]
    list_of_seeds = np.random.uniform(low=1, high=100000, size=(50,)).astype(np.int32)

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

    param_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=[lambdas, xis, list_of_seeds])
    for param_tpl in param_tuples:
        mixing_lambda = param_tpl[0]
        xi = param_tpl[1]
        seed = param_tpl[2]
        np.random.seed(seed)
        network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
        routing_data = DatasetLinkingAlgorithm.link_dataset_v3(network_name_=network_name, run_id_=network_id,
                                                               degree_list_=[2, 2],
                                                               test_iterations_=iterations)
        routing_data.apply_validation_test_split(test_ratio=0.5)
        routing_data.switch_to_single_iteration_mode()
        KmeansPlusBayesianOptimization.optimize(cluster_count=1,
                                                network=network,
                                                routing_data=routing_data,
                                                mixing_lambda=mixing_lambda,
                                                seed=seed,
                                                run_id=network_id,
                                                iteration=0,
                                                xi=xi)
        tf.reset_default_graph()


def prepare_ensemble_data(network_name, iterations, network_ids, output_names):
    list_of_networks = []
    list_of_routing_data = []
    # Prepare the data
    for network_id in network_ids:
        print("Processing network:{0}".format(network_id))
        network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
        routing_data = DatasetLinkingAlgorithm.link_dataset_v3(network_name_=network_name,
                                                               run_id_=network_id,
                                                               degree_list_=[2, 2],
                                                               test_iterations_=iterations,
                                                               output_names=output_names)
        routing_data.switch_to_single_iteration_mode()
        list_of_networks.append(network)
        list_of_routing_data.append(routing_data)
    DatasetLinkingAlgorithm.align_datasets(list_of_datasets=list_of_routing_data,
                                           link_node_index=0,
                                           link_feature="original_samples")
    assert all([np.array_equal(
        list_of_routing_data[idx].dictionaryOfRoutingData["original_samples"][0],
        list_of_routing_data[idx + 1].dictionaryOfRoutingData["original_samples"][0])
                for idx in range(len(list_of_routing_data) - 1)])
    return list_of_networks, list_of_routing_data


def train_test_split_for_ensembles(list_of_routing_data, test_ratio=0.5):
    list_of_routing_data[0].apply_validation_test_split(test_ratio=test_ratio)
    for idx in range(len(list_of_routing_data) - 1):
        list_of_routing_data[idx + 1].trainingIndices = list_of_routing_data[0].trainingIndices
        list_of_routing_data[idx + 1].testIndices = list_of_routing_data[0].testIndices


def train_ensemble_threshold_optimizer():
    # USPS
    network_ids = [2013, 1788]
    run_id = "1731_1788"
    network_name = "USPS_CIGN"
    iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
                         11741, 11800])
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature", "indices_tensor", "original_samples"]
    lambdas = [1.0, 0.99, 0.95, 0.9]
    xis = [0.0, 0.001, 0.01]
    list_of_seeds = np.random.uniform(low=1, high=100000, size=(100,)).astype(np.int32)
    param_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=[lambdas, xis, list_of_seeds])

    list_of_networks, list_of_routing_data = prepare_ensemble_data(network_name=network_name,
                                                                   iterations=iterations,
                                                                   network_ids=network_ids,
                                                                   output_names=output_names)

    # Bayesian Optimization of the ensemble
    for param_tpl in param_tuples:
        mixing_lambda = param_tpl[0]
        xi = param_tpl[1]
        seed = param_tpl[2]
        np.random.seed(seed)
        train_test_split_for_ensembles(list_of_routing_data)
        KmeansPlusBayesianOptimization.optimize_ensemble(
            list_of_networks=list_of_networks,
            list_of_routing_data=list_of_routing_data,
            mixing_lambda=mixing_lambda,
            xi=xi,
            seed=seed,
            run_id=run_id,
            iteration=0)
        tf.reset_default_graph()

        print("X")


def pick_best_ensembles(ensemble_size):
    list_of_network_ids = [1731, 1826, 2013, 1788, 1700, 1995, 1892, 1974, 1973, 1992, 2022, 1699, 1737, 1759, 2054,
                           2036,
                           1918, 1998, 2024, 1963, 2046, 1683, 2055, 1977, 1986, 1724, 1825, 1899, 1851, 1761, 2043,
                           2051,
                           1962, 1860, 1850, 1792, 1957, 1912, 1734, 1893, 1835, 1921, 1844, 1905, 2039, 2038, 1947,
                           1693,
                           2067, 2076, 1971, 1865, 1800, 2065, 1945, 1950, 1786, 1900, 1987, 1870, 1881, 1736, 1990,
                           1842,
                           2048]
    # list_of_network_ids = [1731, 1826, 2013, 1788, 1700]
    network_name = "USPS_CIGN"
    iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
                         11741, 11800])
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature", "indices_tensor", "original_samples"]
    list_of_networks, list_of_routing_data = prepare_ensemble_data(network_name=network_name,
                                                                   iterations=iterations,
                                                                   network_ids=list_of_network_ids,
                                                                   output_names=output_names)
    train_test_split_for_ensembles(list_of_routing_data)
    train_indices = list_of_routing_data[0].trainingIndices
    test_indices = list_of_routing_data[0].testIndices
    network_data_pairs = [tpl for tpl in zip(list_of_network_ids, list_of_networks, list_of_routing_data)]
    network_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=[network_data_pairs] * ensemble_size)
    results = {}
    run_id = DbLogger.get_run_id()
    DbLogger.write_into_table(rows=[(run_id, "Ensemble Picking - Ensemble Count:{0}".format(run_id))],
                              table=DbLogger.runMetaData, col_count=None)
    for tpl in network_tuples:
        unique_network_count = len(set([data_tpl[0] for data_tpl in tpl]))
        if unique_network_count < ensemble_size:
            continue
        ids = tuple([data_tpl[0] for data_tpl in tpl])
        nets = [data_tpl[1] for data_tpl in tpl]
        datasets = [data_tpl[2] for data_tpl in tpl]
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.\
            calculate_for_ensembles(list_of_networks=nets,
                                    list_of_routing_data=datasets,
                                    indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.\
            calculate_for_ensembles(list_of_networks=nets,
                                    list_of_routing_data=datasets,
                                    indices=test_indices)
        A_ = (train_indices.shape[0]) / (train_indices.shape[0] + test_indices.shape[0])
        B_ = (test_indices.shape[0]) / (train_indices.shape[0] + test_indices.shape[0])
        ensemble_accuracy = A_ * train_ig_accuracy + B_ * test_ig_accuracy
        print("{0}: {1}".format(ids, ensemble_accuracy))
        results[ids] = ensemble_accuracy
        DbLogger.write_into_table(rows=[(run_id, 0, "{0}".format(ids), ensemble_accuracy)],
                                  table=DbLogger.runKvStore, col_count=None)
    print(results)
    print("X")


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    # train_direct_threshold_optimizer()
    # train_ensemble_threshold_optimizer()
    pick_best_ensembles(ensemble_size=2)


if __name__ == "__main__":
    main()
