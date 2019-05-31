class SimpleAccuracyCalculator:
    def __init__(self):
        pass

    @staticmethod
    def collect_eval_results_from_network(network,
                                          sess,
                                          dataset,
                                          leaf_node_collection_names,
                                          inner_node_collections_names):
        leaf_node_collections = {}
        inner_node_collections = {}
        while True:
            results, _ = network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            if results is not None:
                batch_sample_count = 0.0
                for node in network.topologicalSortedNodes:















                    if not node.isLeaf:
                        info_gain = results[self.network.get_variable_name(name="info_gain", node=node)]
                        branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                        if GlobalConstants.USE_SAMPLING_CIGN:
                            chosen_indices = results[self.network.get_variable_name(name="chosen_indices", node=node)]
                            UtilityFuncs.concat_to_np_array_dict(dct=chosen_indices_dict, key=node.index,
                                                                 array=chosen_indices)
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs_dict, key=node.index, array=branch_prob)
                        if node.index not in info_gain_dict:
                            info_gain_dict[node.index] = []
                        info_gain_dict[node.index].append(np.asscalar(info_gain))
                        continue
                    if results[self.network.get_variable_name(name="is_open", node=node)] == 0.0:
                        continue
                    posterior_probs = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                    true_labels = results["Node{0}_label_tensor".format(node.index)]
                    final_features = results[self.network.get_variable_name(name="final_feature_final", node=node)]
                    # batch_sample_count += results[self.get_variable_name(name="sample_count", node=node)]
                    predicted_labels = np.argmax(posterior_probs, axis=1)
                    batch_sample_count += predicted_labels.shape[0]
                    UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                                                         array=predicted_labels)
                    UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                         array=true_labels)
                    UtilityFuncs.concat_to_np_array_dict(dct=final_features_dict, key=node.index,
                                                         array=final_features)
                if batch_sample_count != GlobalConstants.EVAL_BATCH_SIZE:
                    raise Exception("Incorrect batch size:{0}".format(batch_sample_count))
            if dataset.isNewEpoch:
                break