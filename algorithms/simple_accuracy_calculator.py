import numpy as np
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants
import os


class SimpleAccuracyCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_accuracy_multipath(network, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        inner_node_outputs = ["p(n|x)", "activations"]
        leaf_node_outputs = ["posterior_probs", "label_tensor"]
        leaf_node_collections, inner_node_collections = \
            network.collect_eval_results_from_network(sess=sess, dataset=dataset, dataset_type=dataset_type,
                                                      use_masking=False,
                                                      leaf_node_collection_names=leaf_node_outputs,
                                                      inner_node_collections_names=inner_node_outputs)
        leaf_true_labels_dict = leaf_node_collections["label_tensor"]
        branch_probs_dict = inner_node_collections["p(n|x)"]
        posterior_probs_dict = leaf_node_collections["posterior_probs"]
        activations_dict = inner_node_collections["activations"]
        # Save the the info from this iteration
        if GlobalConstants.SAVE_PATH_INFO_TO_HD:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            directory_path = os.path.abspath(os.path.join(os.path.join(os.path.join(curr_path, ".."),
                                                                       "saved_training_data"),
                                                          "run_{0}_iteration_{1}".format(run_id, iteration)))
            os.mkdir(directory_path)
            npz_file_name = os.path.abspath(os.path.join(directory_path, "branching_info"))
            UtilityFuncs.save_npz(npz_file_name,
                                  arr_dict={"tree_type": np.array(network.degreeList),
                                            "label_tensor": leaf_true_labels_dict,
                                            "p(n|x)": branch_probs_dict,
                                            "posterior_probs": posterior_probs_dict,
                                            "activations": activations_dict})
        thresholds = []
        for threshold in GlobalConstants.MULTIPATH_SCHEDULES:
            t_dict = {}
            for node in network.topologicalSortedNodes:
                if not node.isLeaf:
                    child_count = len(network.dagObject.children(node=node))
                    t_dict[node.index] = threshold * np.ones(shape=(child_count,))
            thresholds.append(t_dict)
        assert all([np.array_equal(list(leaf_true_labels_dict.values())[0], list(leaf_true_labels_dict.values())[i])
                    for i in range(len(leaf_true_labels_dict))])
        label_list = list(leaf_true_labels_dict.values())[0]
        sample_count = label_list.shape[0]
        multipath_calculator = MultipathCalculatorV2(run_id=run_id, iteration=iteration,
                                                     thresholds_list=thresholds, network=network,
                                                     sample_count=sample_count, label_list=label_list,
                                                     branch_probs=branch_probs_dict,
                                                     activations=activations_dict,
                                                     posterior_probs=posterior_probs_dict)
        results = multipath_calculator.run()
        kv_rows = []
        for result in results:
            # Tuple: Entry 0: Method Entry 1: Thresholds Entry 2: Accuracy Entry 3: Num of leaves evaluated
            # Entry 4: Computation Overload
            method = result[0]
            path_threshold = result[1][0][0]
            accuracy = result[2]
            total_leaves_evaluated = result[4]
            kv_rows.append((run_id, iteration, method, path_threshold, accuracy, total_leaves_evaluated))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.multipath_results_table_v2, col_count=6)
