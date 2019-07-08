import numpy as np
from algorithms.multipath_calculator_v2 import MultipathCalculatorV2
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


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
