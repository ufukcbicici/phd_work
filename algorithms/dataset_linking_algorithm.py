import numpy as np
import os
import pickle
from scipy.spatial import distance_matrix

from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import RoutingDataset, \
    MultiIterationRoutingDataset
from auxillary.db_logger import DbLogger
from simple_tf.cign.fast_tree import FastTreeNetwork
from collections import Counter


# network_id = 1892
# list_of_network_ids = [1731, 1826, 2013, 1788, 1700, 1995, 1892, 1974, 1973, 1992, 2022, 1699, 1737, 1759, 2054, 2036,
#                        1918, 1998, 2024, 1963, 2046, 1683, 2055, 1977, 1986, 1724, 1825, 1899, 1851, 1761, 2043, 2051,
#                        1962, 1860, 1850, 1792, 1957, 1912, 1734, 1893, 1835, 1921, 1844, 1905, 2039, 2038, 1947, 1693,
#                        2067, 2076, 1971, 1865, 1800, 2065, 1945, 1950, 1786, 1900, 1987, 1870, 1881, 1736, 1990, 1842,
#                        2048]
# network_name = "USPS_CIGN"

# output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
#                 "pre_branch_feature", "indices_tensor", "original_samples"]
# used_output_names = ["original_samples"]
# # iterations = sorted([43680, 44160, 44640, 45120, 45600, 46080, 46560, 47040, 47520, 48000])
# iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
#                      11741, 11800])


class DatasetLinkingAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def link_datasets_by_closeness(data_dict, outputs, iteration_list):
        data_length_set = set([data.labelList.shape[0] for data in data_dict.values()])
        assert len(data_length_set) == 1
        sample_count = list(data_length_set)[0]
        # Link all samples among all iterations
        sample_mappings = []
        for sample_id in range(sample_count):
            sample_id_in_iterations = {iteration_list[0]: sample_id}
            print("*************Sample ID:{0}*************".format(sample_id_in_iterations))
            for idx, iteration in enumerate(iteration_list):
                print("*************Iteration:{0}*************".format(iteration))
                sample_id_in_curr_iteration = sample_id_in_iterations[iteration]
                if iteration == iteration_list[-1]:
                    break
                curr_iteration_data = data_dict[iteration_list[idx]]
                next_iteration_data = data_dict[iteration_list[idx + 1]]
                for feature_name in outputs:
                    curr_iteration_feature_dict = curr_iteration_data.get_dict(feature_name)
                    next_iteration_feature_dict = next_iteration_data.get_dict(feature_name)
                    assert set(curr_iteration_feature_dict.keys()) == set(next_iteration_feature_dict.keys())
                    node_ids = sorted(list(curr_iteration_feature_dict.keys()))
                    for node_id in node_ids:
                        print("************************************************************************")
                        X_curr = curr_iteration_feature_dict[node_id]
                        X_next = next_iteration_feature_dict[node_id]
                        X_curr = np.reshape(X_curr, newshape=(X_curr.shape[0], np.prod(X_curr.shape[1:])))
                        X_next = np.reshape(X_next, newshape=(X_next.shape[0], np.prod(X_next.shape[1:])))
                        x_curr = X_curr[sample_id_in_curr_iteration]
                        dif_matrix = x_curr - X_next
                        squared_dif_matrix = np.square(dif_matrix)
                        squared_distances = np.sum(squared_dif_matrix, axis=1)
                        arg_sort_indices = np.argsort(squared_distances)
                        assert squared_distances[arg_sort_indices[0]] < squared_distances[arg_sort_indices[1]]
                        print("SampleId:{0} CurrIteration:{1} NextIteration:{2} FeatureName:{3} NodeId:{4}".format(
                            sample_id, iteration_list[idx], iteration_list[idx + 1], feature_name, node_id))
                        print("arg_sort_indices[0:10]:{0}".format(arg_sort_indices[0:10]))
                        print("squared_distances[arg_sort_indices[0:10]]:{0}".format(
                            squared_distances[arg_sort_indices[0:10]]))
                        sample_id_in_next_iteration = np.asscalar(arg_sort_indices[0])
                        if iteration_list[idx + 1] not in sample_id_in_iterations:
                            sample_id_in_iterations[iteration_list[idx + 1]] = sample_id_in_next_iteration
                        else:
                            assert sample_id_in_iterations[iteration_list[idx + 1]] == sample_id_in_next_iteration
                        x_next = X_next[sample_id_in_next_iteration]
                        assert np.allclose(np.sum(np.square(x_next - x_curr)),
                                           squared_distances[sample_id_in_next_iteration])
                    print("************************************************************************")
            label_set_for_all_iterations = set([data_dict[iteration].labelList[sample_id_in_iterations[iteration]]
                                                for iteration in data_dict.keys()])
            assert len(label_set_for_all_iterations) == 1
            sample_mappings.append(sample_id_in_iterations)
        return sample_mappings

    @staticmethod
    def run(network_id, network_name, iterations, output_names, used_output_names):
        data_dict = {}
        # Load all the data
        for iteration in iterations:
            network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
            routing_data = network.load_routing_info(run_id=network_id, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            data_dict[iteration] = routing_data
        data_length_set = set([data.labelList.shape[0] for data in data_dict.values()])
        assert len(data_length_set) == 1
        sample_mappings = DatasetLinkingAlgorithm.link_datasets_by_closeness(data_dict=data_dict,
                                                                             outputs=used_output_names,
                                                                             iteration_list=iterations)

        db_rows = []
        for sample_id_in_iterations in sample_mappings:
            sample_id = sample_id_in_iterations[iterations[0]]
            for iteration in iterations:
                sample_id_in_curr_iteration = sample_id_in_iterations[iteration]
                for feature_name in used_output_names:
                    for node_id in sorted(list(data_dict[iteration].get_dict(feature_name).keys())):
                        db_rows.append((network_name, network_id, iteration, feature_name, node_id, sample_id,
                                        sample_id_in_curr_iteration))
            DbLogger.write_into_table(rows=db_rows, table="dataset_link", col_count=7)
            db_rows = []

    @staticmethod
    def link_dataset(network_id, network_name, iterations, output_names, used_output_names):
        data_dict = {}
        # Load all the data
        for iteration in iterations:
            network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
            routing_data = network.load_routing_info(run_id=network_id, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            data_dict[iteration] = routing_data
        curr_path = os.path.dirname(os.path.abspath(__file__))
        target_directory = \
            os.path.abspath(os.path.join(curr_path, "..", "saved_training_data",
                                         "{0}_{1}_linked_features".format(network_name, network_id)))
        rows = DbLogger.read_tuples_from_table(table_name="dataset_link")
        iterations_read = sorted(list(set([row[2] for row in rows])))
        feature_names = set([row[3] for row in rows])
        node_ids = set([row[4] for row in rows])
        sample_ids = set([row[5] for row in rows])
        max_sample_id = max([s_id for s_id in sample_ids])
        min_iteration_id = min([iteration for iteration in iterations_read])
        data_dict_read = {}
        for feature_name in feature_names:
            print("feature_name:{0}".format(feature_name))
            arr_dict = data_dict[min_iteration_id].get_dict(feature_name)
            data_dict_read[feature_name] = {}
            for node_id in arr_dict.keys():
                print("node_id:{0}".format(node_id))
                shape_list = list(arr_dict[node_id].shape)
                shape_list.append(len(iterations_read))
                data_dict_read[feature_name][node_id] = np.zeros(shape=tuple(shape_list))
                for s_id in range(max_sample_id + 1):
                    print("s_id:{0}".format(s_id))
                    for idx, iteration_id in enumerate(iterations_read):
                        sample_row = [row for row in rows if
                                      row[0] == network_name and
                                      row[1] == network_id and
                                      row[2] == iteration_id and
                                      row[3] == feature_name and
                                      row[4] == node_id and
                                      row[5] == s_id]
                        assert len(sample_row) == 1
                        s_id_for_iteration = sample_row[0][6]
                        data_dict_read[feature_name][node_id][s_id, ..., idx] = \
                            data_dict[iteration_id].get_dict(feature_name)[node_id][s_id_for_iteration, ...]
        os.mkdir(target_directory)
        chunk_size = 1000
        for feature_name in feature_names:
            for node_id in data_dict_read[feature_name].keys():
                chunk_id = 0
                while True:
                    data_chunk = data_dict_read[feature_name][node_id][
                                 chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
                    if np.prod(data_chunk.shape) == 0:
                        break
                    pickle.dump(data_chunk,
                                open(os.path.abspath(os.path.join(target_directory,
                                                                  "data_dict_read_{0}_node_{1}_chunk_{2}.sav"
                                                                  .format(feature_name, node_id, chunk_id))), "wb"))
                    chunk_id += 1
            # data_dict_read[feature_name] = np.zeros(shape=(arr.shape[0], arr.shape[1], len(iterations_read)))
            # print("X")

            # for sampled_id in range(max_sample_id):
            #     print("X")

        # data_read_dict = {}
        # for row in rows:

        print("X")

    @staticmethod
    def link_dataset_v2(network_name_, run_id_, degree_list_, feature_names_, output_names):
        # Get iterations
        db_iteration_tuples = DbLogger.read_query(
            query="SELECT Iteration, COUNT(1) AS CNT FROM dataset_link "
                  "WHERE NetworkName=\"{0}\" AND NetworkId={1} GROUP BY Iteration".format(network_name_, run_id_))
        db_iterations = sorted([tpl[0] for tpl in db_iteration_tuples])
        assert len(set([tpl[1] for tpl in db_iteration_tuples])) == 1
        # Get iteration mappings
        db_mappings = DbLogger.read_query(
            query="SELECT SampleId, Iteration, SampleIdForIteration, COUNT(1) AS CNT FROM dataset_link "
                  "WHERE NetworkName=\"{0}\" AND NetworkId={1} "
                  "GROUP BY SampleId, Iteration, SampleIdForIteration "
                  "ORDER BY SampleId, Iteration".format(network_name_, run_id_))
        assert len(set([tpl[-1] for tpl in db_mappings])) == 1
        # Load all routing datasets
        routing_data_dict = {}
        # # Load all the data
        for iteration in db_iterations:
            network = FastTreeNetwork.get_mock_tree(degree_list=degree_list_, network_name=network_name_)
            routing_data = network.load_routing_info(run_id=run_id_, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            routing_data_dict[iteration] = routing_data
        # Merge feature tensors
        dict_of_data_dicts = {}
        for feature_name in feature_names_:
            dict_of_data_dicts[feature_name] = {}
            for node_id in routing_data_dict[db_iterations[0]].dictionaryOfRoutingData[feature_name].keys():
                dict_of_data_dicts[feature_name][node_id] = []
        dict_of_data_dicts["nodeCosts"] = {}
        for node_id in routing_data_dict[db_iterations[0]].dictionaryOfRoutingData["nodeCosts"].keys():
            dict_of_data_dicts["nodeCosts"][node_id] = routing_data_dict[db_iterations[0]].dictionaryOfRoutingData[
                "nodeCosts"][node_id]

        merged_labels = []
        # Add features and labels from different iterations into single feature matrices.
        for idx in range(0, len(db_mappings), len(db_iterations)):
            tuples = db_mappings[idx: idx + len(db_iterations)]
            assert len(set([tpl[0] for tpl in tuples])) == 1
            assert len(set([tpl[3] for tpl in tuples])) == 1
            # Get corresponding features from every iteration
            sample_label_list = []
            sample_id_list = []
            for _tpl in tuples:
                sample_id = _tpl[0]
                iteration_id = _tpl[1]
                sample_id_for_iteration = _tpl[2]
                for feature_name in feature_names_:
                    assert set(dict_of_data_dicts[feature_name].keys()) == \
                           set(routing_data_dict[iteration_id].dictionaryOfRoutingData[feature_name].keys())
                    for node_id in dict_of_data_dicts[feature_name].keys():
                        dict_of_data_dicts[feature_name][node_id].append(
                            routing_data_dict[iteration_id].dictionaryOfRoutingData
                            [feature_name][node_id][sample_id_for_iteration])
                sample_label_list.append(routing_data_dict[iteration_id].labelList[sample_id_for_iteration])
                sample_id_list.append(sample_id)
            assert len(set(sample_label_list)) == 1
            assert len(set(sample_label_list)) == 1
            merged_labels.extend(sample_label_list)
            print("Processed Sample:{0}".format(sample_id_list[0]))
        merged_labels = np.array(merged_labels)
        for feature_name in feature_names_:
            for node_id in dict_of_data_dicts[feature_name].keys():
                if feature_name == "nodeCosts":
                    continue
                print("Proccessing feature:{0} node:{1}".format(feature_name, node_id))
                dict_of_data_dicts[feature_name][node_id] = np.stack(dict_of_data_dicts[feature_name][node_id], axis=0)
        routing_dataset = RoutingDataset(dict_of_data_dicts=dict_of_data_dicts, label_list=merged_labels,
                                         index_multiplier=len(db_iterations))
        return routing_dataset

    @staticmethod
    def link_dataset_v3(network_name_, run_id_, degree_list_, test_iterations_, output_names):
        # Get iterations
        db_iteration_tuples = DbLogger.read_query(
            query="SELECT Iteration, COUNT(1) AS CNT FROM dataset_link "
                  "WHERE NetworkName=\"{0}\" AND NetworkId={1} GROUP BY Iteration".format(network_name_, run_id_))
        db_iterations = sorted([tpl[0] for tpl in db_iteration_tuples])
        assert len(set([tpl[1] for tpl in db_iteration_tuples])) == 1
        # Get iteration mappings
        db_mappings = DbLogger.read_query(
            query="SELECT SampleId, Iteration, SampleIdForIteration, COUNT(1) AS CNT FROM dataset_link "
                  "WHERE NetworkName=\"{0}\" AND NetworkId={1} "
                  "GROUP BY SampleId, Iteration, SampleIdForIteration "
                  "ORDER BY SampleId, Iteration".format(network_name_, run_id_))
        assert len(set([tpl[-1] for tpl in db_mappings])) == 1
        # Load all routing datasets
        routing_data_dict = {}
        # # Load all the data
        for iteration in db_iterations:
            network = FastTreeNetwork.get_mock_tree(degree_list=degree_list_, network_name=network_name_)
            routing_data = network.load_routing_info(run_id=run_id_, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            routing_data_dict[iteration] = routing_data
        multipath_routing_dataset = MultiIterationRoutingDataset(
            dict_of_routing_datasets=routing_data_dict, sample_linkage_info=db_mappings,
            test_iterations=test_iterations_)
        return multipath_routing_dataset

    @staticmethod
    def align_datasets(list_of_datasets, link_node_index, link_feature):
        assert len(set([dataset.singleDatasetSize for dataset in list_of_datasets])) == 1
        reference_dataset = list_of_datasets[0]
        reference_features = reference_dataset.get_dict(link_feature)[link_node_index]
        reference_feature_sums_as_hash_values = np.sum(reference_features, axis=1)
        array_of_indices = []
        ignored_samples_for_all_datasets = set()
        # Reverse index of features
        for dataset in list_of_datasets:
            print("Dataset")
            ignored_samples = set()
            multiple_hit_count = 0
            features = dataset.get_dict(link_feature)[link_node_index]
            dataset_reverse_index = {}
            feature_sums_as_hash_values = np.sum(features, axis=1)
            for idx in range(feature_sums_as_hash_values.shape[0]):
                feature_hash = feature_sums_as_hash_values[idx]
                iteration = idx // int(reference_dataset.singleDatasetSize)
                if (feature_hash, iteration) in dataset_reverse_index:
                    # assert dataset.labelList[
                    #     dataset_reverse_index[(feature_hash, iteration)]] == dataset.labelList[idx]
                    ignored_samples.add(feature_hash)
                    # print("Exists!")
                    multiple_hit_count += 1
                else:
                    dataset_reverse_index[(feature_hash, iteration)] = idx
            array_of_indices.append(dataset_reverse_index)
            print("multiple_hit_count={0}".format(multiple_hit_count))
            ignored_samples_for_all_datasets.add(frozenset(ignored_samples))
        assert len(ignored_samples_for_all_datasets) == 1
        ignored_samples = list(ignored_samples_for_all_datasets)[0]
        print("X")
        # Align all other datasets according to the first one. This means i.th feature in the arrays of the first
        # dataset corresponds to the i.th feature to all other arrays of the datasets.
        for d_id in range(len(list_of_datasets)):
            dataset_sample_index = array_of_indices[d_id]
            reloc_indices = []
            for s_id in range(reference_feature_sums_as_hash_values.shape[0]):
                hash_code = reference_feature_sums_as_hash_values[s_id]
                if hash_code in ignored_samples:
                    continue
                iteration = s_id // int(reference_dataset.singleDatasetSize)
                reloc_indices.append(dataset_sample_index[(hash_code, iteration)])
            # Realign the other dataset
            reloc_indices = np.array(reloc_indices)
            other_dataset = list_of_datasets[d_id]
            other_dataset.labelList = other_dataset.labelList[reloc_indices]
            other_dataset.linkageInfo = reference_dataset.linkageInfo
            for feature_name in other_dataset.dictionaryOfRoutingData.keys():
                if "Cost" in feature_name or "cost" in feature_name:
                    continue
                feature_dict = other_dataset.dictionaryOfRoutingData[feature_name]
                for node_id in feature_dict.keys():
                    other_dataset.dictionaryOfRoutingData[feature_name][node_id] = feature_dict[node_id][reloc_indices]
        # Sanity check for labels
        for d_id in range(len(list_of_datasets) - 1):
            assert np.array_equal(list_of_datasets[d_id].labelList, list_of_datasets[d_id + 1].labelList)
        for d_id in range(len(list_of_datasets)):
            list_of_datasets[d_id].singleDatasetSize -= len(ignored_samples)


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    # 16-12-8 networks, without Bootstrapping
    # list_of_network_ids = [1731, 1826, 2013, 1788, 1700, 1995, 1892, 1974, 1973, 1992, 2022, 1699, 1737, 1759, 2054,
    #                        2036,
    #                        1918, 1998, 2024, 1963, 2046, 1683, 2055, 1977, 1986, 1724, 1825, 1899, 1851, 1761, 2043,
    #                        2051,
    #                        1962, 1860, 1850, 1792, 1957, 1912, 1734, 1893, 1835, 1921, 1844, 1905, 2039, 2038, 1947,
    #                        1693,
    #                        2067, 2076, 1971, 1865, 1800, 2065, 1945, 1950, 1786, 1900, 1987, 1870, 1881, 1736, 1990,
    #                        1842,
    #                        2048]
    # 64-32-16 networks, with BootStrapping
    # list_of_network_ids = [863, 836, 792, 776, 768, 748, 720, 700, 683, 681, 617, 588, 573, 569, 554, 540, 539,
    #                        523, 516, 502, 494, 478, 456, 451, 447, 445]

    list_of_network_ids = [350, 390, 421, 426, 352, 329, 295, 333, 319]
    network_name = "USPS_CIGN"
    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature", "indices_tensor", "original_samples"]
    used_output_names = ["original_samples"]
    # iterations = sorted([43680, 44160, 44640, 45120, 45600, 46080, 46560, 47040, 47520, 48000])
    iterations = sorted([10974, 11033, 11092, 11151, 11210, 11269, 11328, 11387, 11446, 11505, 11564, 11623, 11682,
                         11741, 11800])
    for network_id in list_of_network_ids:
        DatasetLinkingAlgorithm.run(network_id=network_id,
                                    network_name=network_name,
                                    iterations=iterations, output_names=output_names,
                                    used_output_names=used_output_names)


if __name__ == "__main__":
    main()
