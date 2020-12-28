import numpy as np
import tensorflow as tf
import os
from algorithms.dataset_linking_algorithm import DatasetLinkingAlgorithm
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.kmeans_plus_bayesian_optimization import \
    KmeansPlusBayesianOptimization
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from auxillary.db_logger import DbLogger
from simple_tf.global_params import GlobalConstants
from sklearn.model_selection import train_test_split


def train_direct_threshold_optimizer():
    # Fashion MNIST
    # network_id = 453
    # network_name = "FashionNet_Lite"
    # iteration = 47520
    # list_of_l2_coeffs = [0.0, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001,
    #                      0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005, 0.01]

    # USPS
    network_id = 1800
    network_name = "USPS_CIGN"
    iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
                         11741, 11800])

    # lambdas = [1.0, 0.99, 0.95, 0.9]
    # xis = [0.0, 0.001, 0.01]
    lambdas = [1.0]
    xis = [0.0]
    list_of_seeds = np.random.uniform(low=1, high=100000, size=(50,)).astype(np.int32)

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]

    param_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=[lambdas, xis, list_of_seeds])
    for param_tpl in param_tuples:
        mixing_lambda = param_tpl[0]
        xi = param_tpl[1]
        seed = param_tpl[2]
        np.random.seed(seed)
        network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
        routing_data = DatasetLinkingAlgorithm.link_dataset_v3(network_name_=network_name,
                                                               run_id_=network_id,
                                                               degree_list_=[2, 2],
                                                               test_iterations_=iterations,
                                                               output_names=output_names)
        routing_data.switch_to_single_iteration_mode()
        routing_data.apply_validation_test_split(test_ratio=0.5)
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
    # (1800, 1683, 2048, 1992, 1786, 2076)
    network_ids = [1800, 1683, 2048, 1992, 1786, 2076]
    run_id = "(1800, 1683, 2048, 1992, 1786, 2076)"
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
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator. \
            calculate_for_ensembles(list_of_networks=nets,
                                    list_of_routing_data=datasets,
                                    indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator. \
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


def beam_search_ensembles(max_ensemble_size, beam_size):
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
    dict_of_networks = {n_id: net for n_id, net in zip(list_of_network_ids, list_of_networks)}
    dict_of_data = {n_id: data for n_id, data in zip(list_of_network_ids, list_of_routing_data)}

    selected_tuples = {(n_id,): InformationGainRoutingAccuracyCalculator.calculate_for_ensembles(
        list_of_networks=[dict_of_networks[n_id]],
        list_of_routing_data=[dict_of_data[n_id]],
        indices=np.arange(dict_of_data[n_id].labelList.shape[0])) for n_id in list_of_network_ids}

    run_id = DbLogger.get_run_id()
    DbLogger.write_into_table(rows=[(run_id, "Ensemble Beam Search - Ensemble Count:{0}".format(run_id))],
                              table=DbLogger.runMetaData, col_count=None)

    for ensemble_id in range(max_ensemble_size):
        results_dict = {}
        processed_ensembles = set()
        for n_id in list_of_network_ids:
            for ensemble_members in selected_tuples.keys():
                ensemble = list(ensemble_members)
                ensemble.append(n_id)
                if len(ensemble) > len(set(ensemble)):
                    continue
                ensemble_set = frozenset(ensemble)
                if ensemble_set in processed_ensembles:
                    continue
                processed_ensembles.add(ensemble_set)
                ensemble_networks = [dict_of_networks[e_id] for e_id in ensemble]
                ensemble_data = [dict_of_data[e_id] for e_id in ensemble]
                indices = np.arange(dict_of_data[ensemble[0]].labelList.shape[0])
                accuracy = InformationGainRoutingAccuracyCalculator. \
                    calculate_for_ensembles(list_of_networks=ensemble_networks,
                                            list_of_routing_data=ensemble_data,
                                            indices=indices)
                results_dict[tuple(ensemble)] = accuracy
                print("{0}={1}".format(ensemble, accuracy))
                DbLogger.write_into_table(rows=[(run_id, 0, "{0}".format(tuple(ensemble)), accuracy)],
                                          table=DbLogger.runKvStore, col_count=None)
        sorted_results = sorted([(k, v) for k, v in results_dict.items()], key=lambda tpl: tpl[1], reverse=True)
        beam_results = sorted_results[:beam_size]
        selected_tuples = {tpl[0]: tpl[1] for tpl in beam_results}
    print("Selected Ensembles:{0}".format(selected_tuples))
    print("X")


def multi_bayesian_optimization(network_name, trial_count, iterations,
                                list_of_network_ids, output_names, lambdas, xis):
    list_of_networks, list_of_routing_data = prepare_ensemble_data(network_name=network_name,
                                                                   iterations=iterations,
                                                                   network_ids=list_of_network_ids,
                                                                   output_names=output_names)
    save_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                    GlobalConstants.MODEL_SAVE_FOLDER)
    for network_id, network, routing_data in zip(list_of_network_ids, list_of_networks, list_of_routing_data):
        param_tuples = UtilityFuncs.get_cartesian_product(list_of_lists=[lambdas, xis])
        param_tuples = param_tuples * trial_count
        trial_id = 0
        for param_tpl in param_tuples:
            mixing_lambda = param_tpl[0]
            xi = param_tpl[1]
            seed = int(np.random.uniform(low=1, high=1000000))
            # routing_data.apply_validation_test_split(test_ratio=0.1)
            routing_data.trainingIndices, routing_data.testIndices = \
                train_test_split(np.arange(routing_data.labelList.shape[0]), test_size=0.1)
            best_results = KmeansPlusBayesianOptimization.optimize(cluster_count=1,
                                                                   network=network,
                                                                   routing_data=routing_data,
                                                                   mixing_lambda=mixing_lambda,
                                                                   seed=seed,
                                                                   run_id=network_id,
                                                                   iteration=0,
                                                                   xi=xi)
            save_file_name = "bo_net_{0}_thrs_{1}.sav".format(network_id, trial_id)
            save_file_path = os.path.join(save_folder_path, save_file_name)
            UtilityFuncs.pickle_save_to_file(path=save_file_path, file_content=best_results)
            trial_id += 1
            tf.reset_default_graph()


def bayesian_ensembling(list_of_network_ids, ensemble_size, max_search_count, single_search_size):
    run_id = DbLogger.get_run_id()
    DbLogger.write_into_table(rows=[(run_id, "Best Ensemble Search After Bayesian Optimization:{0}".format(run_id))],
                              table=DbLogger.runMetaData, col_count=None)
    DbLogger.write_into_table(rows=[(run_id, 0, "Ensemble Count", ensemble_size)],
                              table=DbLogger.runKvStore, col_count=None)
    # Read relevant Bayesian Optimization results

    save_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                    GlobalConstants.MODEL_SAVE_FOLDER)
    file_list = os.listdir(save_folder_path)
    bo_results_dict = {n_id: {} for n_id in list_of_network_ids}
    for network_id in list_of_network_ids:
        related_files = [file_name for file_name in file_list if "bo_net_{0}".format(network_id) in file_name]
        for bo_result_file_name in related_files:
            save_file_path = os.path.join(save_folder_path, bo_result_file_name)
            result_dict = UtilityFuncs.pickle_load_from_file(path=save_file_path)
            threshold_id = int(bo_result_file_name.split("_")[-1].split(".")[0])
            assert threshold_id not in bo_results_dict[network_id]
            bo_results_dict[network_id][threshold_id] = result_dict
    # Check if all ground truth labels are the same
    labels_arr = []
    for d1 in bo_results_dict.values():
        for d2 in d1.values():
            labels_arr.append(d2["gt_labels"])
    labels_matrix = np.stack(labels_arr, axis=-1)
    equality_matrix = labels_matrix == labels_matrix[:, 0][:, np.newaxis]
    assert np.all(equality_matrix)
    ground_truth_labels = labels_matrix[:, 0]
    # Create ensembles out of all possible results
    used_combinations_set = set()
    for iteration_id in range(max_search_count):
        # Pick random networks (without replacement in every ensemble)
        # Pick a threshold for every network
        posteriors = []
        activation_costs = []
        for idx in range(single_search_size):
            ensemble_ids = np.random.choice(list_of_network_ids, ensemble_size, False)
            threshold_ids = [np.random.choice(list(bo_results_dict[n_id].keys()), 1)[0] for n_id in ensemble_ids]
            ensemble_id_matrix = np.stack([ensemble_ids, threshold_ids], axis=-1)
            ensemble_code = [tuple(ensemble_id_matrix[r].tolist()) for r in range(ensemble_id_matrix.shape[0])]
            ensemble_code = frozenset(ensemble_code)
            # ensemble_code = np.reshape(ensemble_code, newshape=(ensemble_code.shape[0] * ensemble_code.shape[1],))
            # ensemble_code = tuple(ensemble_code.tolist())
            if len(ensemble_code) < ensemble_size or ensemble_code in used_combinations_set:
                continue
            ensemble_posteriors = []
            ensemble_activation_costs = []
            for tpl in ensemble_code:
                n_id = tpl[0]
                t_id = tpl[1]
                p_ = bo_results_dict[n_id][t_id]["final_posteriors"]
                ensemble_posteriors.append(p_)
                activation_cost = bo_results_dict[n_id][t_id]["final_activation_cost"]
                ensemble_activation_costs.append(activation_cost)
            ensemble_posteriors = np.stack(ensemble_posteriors, axis=-1)
            posteriors.append(ensemble_posteriors)
            ensemble_activation_costs = np.array(ensemble_activation_costs)
            activation_costs.append(ensemble_activation_costs)
            used_combinations_set.add(ensemble_code)
        if len(posteriors) == 0:
            continue
        posteriors_tensor = np.stack(posteriors, axis=0)
        posteriors_averaged = np.mean(posteriors_tensor, axis=-1)
        predictions_matrix = np.argmax(posteriors_averaged, axis=-1)
        activation_costs_matrix = np.stack(activation_costs, axis=0)
        activation_costs_averaged = np.mean(activation_costs_matrix, axis=-1)
        # Do the accuracy calculation
        comparison_matrix = predictions_matrix == ground_truth_labels[np.newaxis, :]
        accuracies = np.mean(comparison_matrix, axis=-1)
        best_accuracy_idx = np.argmax(accuracies)
        best_accuracy = accuracies[best_accuracy_idx]
        best_activation_cost = activation_costs_averaged[best_accuracy_idx]
        print("Iteration:{0} Best Accuracy:{1} Best Activation Cost:{2}".format(iteration_id, best_accuracy,
                                                                                best_activation_cost))
        DbLogger.write_into_table(rows=[(run_id, 0, "Best Accuracy", np.asscalar(best_accuracy))],
                                  table=DbLogger.runKvStore, col_count=None)
        DbLogger.write_into_table(rows=[(run_id, 0, "Best Activation Cost", np.asscalar(best_activation_cost))],
                                  table=DbLogger.runKvStore, col_count=None)
        print("X")


def bayesian_ensembling_exhaustive(list_of_network_ids, ensemble_size, single_search_size):
    run_id = DbLogger.get_run_id()
    DbLogger.write_into_table(rows=[(run_id, "Exhaustive Ensemble Search After Bayesian Optimization:{0}"
                                     .format(run_id))],
                              table=DbLogger.runMetaData, col_count=None)
    DbLogger.write_into_table(rows=[(run_id, 0, "Ensemble Count", ensemble_size)],
                              table=DbLogger.runKvStore, col_count=None)
    # Read relevant Bayesian Optimization results

    save_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                    GlobalConstants.MODEL_SAVE_FOLDER)
    file_list = os.listdir(save_folder_path)
    bo_results_dict = {n_id: {} for n_id in list_of_network_ids}
    for network_id in list_of_network_ids:
        related_files = [file_name for file_name in file_list if "bo_net_{0}".format(network_id) in file_name]
        for bo_result_file_name in related_files:
            save_file_path = os.path.join(save_folder_path, bo_result_file_name)
            result_dict = UtilityFuncs.pickle_load_from_file(path=save_file_path)
            threshold_id = int(bo_result_file_name.split("_")[-1].split(".")[0])
            assert threshold_id not in bo_results_dict[network_id]
            bo_results_dict[network_id][threshold_id] = result_dict
    # Check if all ground truth labels are the same
    labels_arr = []
    for d1 in bo_results_dict.values():
        for d2 in d1.values():
            labels_arr.append(d2["gt_labels"])
    labels_matrix = np.stack(labels_arr, axis=-1)
    equality_matrix = labels_matrix == labels_matrix[:, 0][:, np.newaxis]
    assert np.all(equality_matrix)
    ground_truth_labels = labels_matrix[:, 0]
    # Create ensembles out of all possible results
    used_combinations_set = set()
    network_threshold_pairs = []
    for n_id, d1 in bo_results_dict.items():
        for t_id in d1.keys():
            network_threshold_pairs.append((n_id, t_id))
    all_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[network_threshold_pairs] * ensemble_size)
    # valid_combinations = [comb for comb in all_combinations if len(set(comb)) == ensemble_size]
    combination_buffer = []
    for comb_id, combination in enumerate(all_combinations):
        combination_buffer.append(combination)
        if len(combination_buffer) < single_search_size:
            continue
        posteriors = []
        activation_costs = []
        for comb_ in combination_buffer:
            ensemble_posteriors = []
            ensemble_activation_costs = []
            for tpl in comb_:
                n_id = tpl[0]
                t_id = tpl[1]
                p_ = bo_results_dict[n_id][t_id]["final_posteriors"]
                ensemble_posteriors.append(p_)
                activation_cost = bo_results_dict[n_id][t_id]["final_activation_cost"]
                ensemble_activation_costs.append(activation_cost)
            ensemble_posteriors = np.stack(ensemble_posteriors, axis=-1)
            posteriors.append(ensemble_posteriors)
            ensemble_activation_costs = np.array(ensemble_activation_costs)
            activation_costs.append(ensemble_activation_costs)
        posteriors_tensor = np.stack(posteriors, axis=0)
        posteriors_averaged = np.mean(posteriors_tensor, axis=-1)
        predictions_matrix = np.argmax(posteriors_averaged, axis=-1)
        activation_costs_matrix = np.stack(activation_costs, axis=0)
        activation_costs_averaged = np.mean(activation_costs_matrix, axis=-1)
        # Do the accuracy calculation
        comparison_matrix = predictions_matrix == ground_truth_labels[np.newaxis, :]
        accuracies = np.mean(comparison_matrix, axis=-1)
        best_accuracy_idx = np.argmax(accuracies)
        best_accuracy = accuracies[best_accuracy_idx]
        best_activation_cost = activation_costs_averaged[best_accuracy_idx]
        print("Iteration:{0} Best Accuracy:{1} Best Activation Cost:{2}".format(comb_id, best_accuracy,
                                                                                best_activation_cost))
        DbLogger.write_into_table(rows=[(run_id, 0, "Best Accuracy", np.asscalar(best_accuracy))],
                                  table=DbLogger.runKvStore, col_count=None)
        DbLogger.write_into_table(rows=[(run_id, 0, "Best Activation Cost", np.asscalar(best_activation_cost))],
                                  table=DbLogger.runKvStore, col_count=None)
        combination_buffer = []
        print("X")


def main():
    # list_of_network_ids = [1731, 1826, 2013, 1788, 1700, 1995, 1892, 1974, 1973, 1992, 2022, 1699, 1737, 1759, 2054,
    #                        2036,
    #                        1918, 1998, 2024, 1963, 2046, 1683, 2055, 1977, 1986, 1724, 1825, 1899, 1851, 1761, 2043,
    #                        2051,
    #                        1962, 1860, 1850, 1792, 1957, 1912, 1734, 1893, 1835, 1921, 1844, 1905, 2039, 2038, 1947,
    #                        1693,
    #                        2067, 2076, 1971, 1865, 1800, 2065, 1945, 1950, 1786, 1900, 1987, 1870, 1881, 1736, 1990,
    #                        1842,
    #                        2048]
    # list_of_network_ids = [1731, 1826, 2013, 1788, 1700]
    list_of_network_ids = [350, 390, 421, 426, 352, 329, 295, 333]
    network_name = "USPS_CIGN"
    iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
                         11741, 11800])
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature", "indices_tensor", "original_samples"]
    lambdas = [1.0]
    xis = [0.0]

    # compare_gpu_implementation()
    # train_basic_q_learning()
    # train_direct_threshold_optimizer()
    # train_ensemble_threshold_optimizer()
    # pick_best_ensembles(ensemble_size=2)
    # beam_search_ensembles(max_ensemble_size=10, beam_size=65)
    multi_bayesian_optimization(network_name=network_name,
                                list_of_network_ids=list_of_network_ids,
                                trial_count=25,
                                iterations=iterations,
                                lambdas=lambdas,
                                xis=xis,
                                output_names=output_names)
    # bayesian_ensembling(list_of_network_ids=list_of_network_ids,
    #                     ensemble_size=3,
    #                     max_search_count=10000,
    #                     single_search_size=100)
    # bayesian_ensembling_exhaustive(list_of_network_ids=list_of_network_ids, ensemble_size=3, single_search_size=100)


if __name__ == "__main__":
    main()
