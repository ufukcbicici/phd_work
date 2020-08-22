import numpy as np
import os
import pickle
from scipy.spatial import distance_matrix

from algorithms.threshold_optimization_algorithms.threshold_optimization_helpers import RoutingDataset, \
    MultiIterationRoutingDataset
from auxillary.db_logger import DbLogger
from simple_tf.cign.fast_tree import FastTreeNetwork

network_id = 453
network_name = "FashionNet_Lite"

output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                "pre_branch_feature"]
used_output_names = ["pre_branch_feature"]
iterations = sorted([43680, 44160, 44640, 45120, 45600, 46080, 46560, 47040, 47520, 48000])


class DatasetLinkingAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def run():
        data_dict = {}
        # Load all the data
        for iteration in iterations:
            network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
            routing_data = network.load_routing_info(run_id=network_id, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            data_dict[iteration] = routing_data
        data_length_set = set([data.labelList.shape[0] for data in data_dict.values()])
        assert len(data_length_set) == 1
        sample_count = list(data_length_set)[0]
        # Link all samples among all iterations
        for sample_id in range(sample_count):
            sample_id_in_iterations = {iterations[0]: sample_id}
            print("*************Sample ID:{0}*************".format(sample_id_in_iterations))
            for idx, iteration in enumerate(iterations):
                print("*************Iteration:{0}*************".format(iteration))
                sample_id_in_curr_iteration = sample_id_in_iterations[iteration]
                if iteration == iterations[-1]:
                    break
                curr_iteration_data = data_dict[iterations[idx]]
                next_iteration_data = data_dict[iterations[idx + 1]]
                for feature_name in used_output_names:
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
                            sample_id, iterations[idx], iterations[idx + 1], feature_name, node_id))
                        print("arg_sort_indices[0:10]:{0}".format(arg_sort_indices[0:10]))
                        print("squared_distances[arg_sort_indices[0:10]]:{0}".format(
                            squared_distances[arg_sort_indices[0:10]]))
                        sample_id_in_next_iteration = np.asscalar(arg_sort_indices[0])
                        if iterations[idx + 1] not in sample_id_in_iterations:
                            sample_id_in_iterations[iterations[idx + 1]] = sample_id_in_next_iteration
                        else:
                            assert sample_id_in_iterations[iterations[idx + 1]] == sample_id_in_next_iteration
                        x_next = X_next[sample_id_in_next_iteration]
                        assert np.allclose(np.sum(np.square(x_next - x_curr)),
                                           squared_distances[sample_id_in_next_iteration])
                    print("************************************************************************")
            label_set_for_all_iterations = set([data_dict[iteration].labelList[sample_id_in_iterations[iteration]]
                                                for iteration in data_dict.keys()])
            assert len(label_set_for_all_iterations) == 1
            db_rows = []
            for iteration in iterations:
                sample_id_in_curr_iteration = sample_id_in_iterations[iteration]
                for feature_name in used_output_names:
                    for node_id in sorted(list(data_dict[iteration].get_dict(feature_name).keys())):
                        db_rows.append((network_name, network_id, iteration, feature_name, node_id, sample_id,
                                        sample_id_in_curr_iteration))
            DbLogger.write_into_table(rows=db_rows, table="dataset_link", col_count=7)

    @staticmethod
    def link_dataset():
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
    def link_dataset_v2(network_name_, run_id_, degree_list_, feature_names_):
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
    def link_dataset_v3(network_name_, run_id_, degree_list_, test_iterations_):
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


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    # DatasetLinkingAlgorithm.run()
    DatasetLinkingAlgorithm.link_dataset_v3(network_name_="FashionNet_Lite", run_id_=453, degree_list_=[2, 2],
                                            test_iterations_=[48000])


if __name__ == "__main__":
    main()
