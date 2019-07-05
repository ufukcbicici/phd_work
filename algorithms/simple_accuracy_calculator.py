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
        # This is temporary
        thread_count = 1
        threshold_dict = UtilityFuncs.distribute_evenly_to_threads(
            num_of_threads=thread_count,
            list_to_distribute=thresholds)
        threads_dict = {}
        for thread_id in range(thread_count):
            threads_dict[thread_id] = MultipathCalculatorV2(thread_id=thread_id, run_id=run_id, iteration=iteration,
                                                            thresholds_dict=threshold_dict[thread_id], network=network,
                                                            sample_count=sample_count, label_list=label_list,
                                                            branch_probs=branch_probs_dict,
                                                            posterior_probs=posterior_probs_dict)
            threads_dict[thread_id].start()
        all_results = []
        for thread in threads_dict.values():
            thread.join()
        for thread in threads_dict.values():
            all_results.extend(thread.kvRows)
        DbLogger.write_into_table(rows=all_results, table=DbLogger.multipath_results_table_v2, col_count=6)
