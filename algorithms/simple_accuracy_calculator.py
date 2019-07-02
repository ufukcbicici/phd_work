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






        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        branch_activations = {}
        branch_probs = {}
        one_hot_branch_probs = {}
        posterior_probs = {}
        hash_codes = {}