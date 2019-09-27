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
        # SimpleAccuracyCalculator.test_save_load(network=network, run_id=run_id, iteration=iteration,
        #                                         leaf_true_labels_dict=leaf_true_labels_dict,
        #                                         branch_probs_dict=branch_probs_dict,
        #                                         posterior_probs_dict=posterior_probs_dict,
        #                                         activations_dict=activations_dict)
        # # Save the the info from this iteration
        if GlobalConstants.SAVE_PATH_INFO_TO_HD:
            SimpleAccuracyCalculator.save_routing_info(network=network, run_id=run_id, iteration=iteration,
                                                       leaf_true_labels_dict=leaf_true_labels_dict,
                                                       branch_probs_dict=branch_probs_dict,
                                                       posterior_probs_dict=posterior_probs_dict,
                                                       activations_dict=activations_dict)
        thresholds = []
        for threshold in GlobalConstants.MULTIPATH_SCHEDULES:
            t_dict = {}
            for node in network.topologicalSortedNodes:
                if not node.isLeaf:
                    child_count = len(network.dagObject.children(node=node))
                    t_dict[node.index] = threshold * np.ones(shape=(child_count,))
            thresholds.append(t_dict)
        leaf_labels = list(leaf_true_labels_dict.values())
        assert all([np.array_equal(leaf_labels[0], leaf_labels[i]) for i in range(len(leaf_labels))])
        label_list = leaf_labels[0]
        sample_count = label_list.shape[0]
        multipath_calculator = MultipathCalculatorV2(thresholds_list=thresholds, network=network,
                                                     sample_count=sample_count, label_list=label_list,
                                                     branch_probs=branch_probs_dict,
                                                     activations=activations_dict,
                                                     posterior_probs=posterior_probs_dict)
        results = multipath_calculator.run()
        kv_rows = []
        for result in results:
            # Tuple: Entry 0: Method Entry 1: Thresholds Entry 2: Accuracy Entry 3: Num of leaves evaluated
            # Entry 4: Computation Overload
            method = result.methodType
            path_threshold = result.thresholdsDict[0][0]
            accuracy = result.accuracy
            total_leaves_evaluated = result.totalLeavesEvaluated
            kv_rows.append((run_id, iteration, method, path_threshold, accuracy, total_leaves_evaluated))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.multipath_results_table_v2, col_count=6)
