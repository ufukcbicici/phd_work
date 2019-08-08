import numpy as np
import time
from algorithms.multipath_calculator import MultipathCalculator
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class AccuracyCalculator:
    def __init__(self, network):
        self.network = network
        self.modesHistory = []

    @staticmethod
    def measure_branch_probs(branch_probs, run_id, iteration, dataset_type, kv_rows):
        # Measure Branching Probabilities
        for k, v in branch_probs.items():
            p_n = np.mean(v, axis=0)
            arg_max_arr = np.argmax(v, axis=1)
            max_counts = {i: np.sum(arg_max_arr == i) for i in range(p_n.shape[0])}
            print("Argmax counts:{0}".format(max_counts))
            for branch in range(p_n.shape[0]):
                print("{0} p_{1}({2})={3}".format(dataset_type, k, branch, p_n[branch]))
                kv_rows.append((run_id, iteration, "{0} p_{1}({2})".format(dataset_type, k, branch),
                                np.asscalar(p_n[branch])))

    def prepare_sample_wise_statistics(self, run_id, iteration, hash_list, sample_count, branch_probs,
                                       branch_activations, posterior_probs, true_labels):
        kv_rows = []
        for sample_index in range(sample_count):
            sample_label = true_labels[sample_index]
            hash_code = "{0}".format(hash_list[sample_index])
            kv_rows.append((run_id,
                            iteration,
                            hash_code,
                            "Label",
                            np.asscalar(sample_label)))
            for node in self.network.topologicalSortedNodes:
                if not node.isLeaf:
                    branch_prob = branch_probs[node.index][sample_index, :]
                    branch_activation = branch_activations[node.index][sample_index, :]
                    for i in range(branch_prob.shape[0]):
                        kv_rows.append(
                            (run_id,
                             iteration,
                             hash_code,
                             "branch_p_{0}_({1})".format(node.index, i),
                             np.asscalar(branch_prob[i])
                             ))
                        kv_rows.append(
                            (run_id,
                             iteration,
                             hash_code,
                             "branch_a_{0}_({1})".format(node.index, i),
                             np.asscalar(branch_activation[i])
                             ))
                else:
                    posterior = posterior_probs[node.index][sample_index, :]
                    for i in range(posterior.shape[0]):
                        kv_rows.append(
                            (run_id,
                             iteration,
                             hash_code,
                             "posterior_p_{0}_({1})".format(node.index, i),
                             np.asscalar(posterior[i])
                             ))
        return kv_rows

    def label_distribution_analysis(self,
                                    run_id,
                                    iteration,
                                    kv_rows,
                                    leaf_true_labels_dict,
                                    dataset,
                                    dataset_type):
        label_count = dataset.get_label_count()
        label_distribution = np.zeros(shape=(label_count,))
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if node.index not in leaf_true_labels_dict:
                continue
            true_labels = leaf_true_labels_dict[node.index]
            for l in range(label_count):
                label_distribution[l] = np.sum(true_labels == l)
                kv_rows.append((run_id, iteration, "{0} Leaf:{1} True Label:{2}".
                                format(dataset_type, node.index, l), np.asscalar(label_distribution[l])))

    def calculate_accuracy(self, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        final_features_dict = {}
        info_gain_dict = {}
        branch_probs_dict = {}
        chosen_indices_dict = {}
        t0 = time.time()
        while True:
            results, _ = self.network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            if results is not None:
                batch_sample_count = 0.0
                for node in self.network.topologicalSortedNodes:
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
        t1 = time.time()
        print(t1 - t0)
        print("****************Dataset:{0}****************".format(dataset_type))
        if GlobalConstants.USE_SAMPLING_CIGN:
            for node_id in branch_probs_dict.keys():
                _p = np.argmax(branch_probs_dict[node_id], axis=1)
                _q = chosen_indices_dict[node_id]
                assert np.array_equal(_p, _q)
        kv_rows = []
        # Measure final feature statistics
        for k, v in final_features_dict.items():
            max_feature_entry = np.max(v)
            min_feature_entry = np.min(v)
            feature_magnitudes = np.array([np.linalg.norm(v[i, :]) for i in range(v.shape[0])])
            mean_feature_magnitude = np.mean(feature_magnitudes)
            max_feature_magnitude = np.max(feature_magnitudes)
            min_feature_magnitude = np.min(feature_magnitudes)
            std_feature_magnitude = np.std(feature_magnitudes)
            # kv_rows.append((run_id, iteration, "Leaf {0} {1} max_feature_entry".format(k, dataset_type),
            #                 np.asscalar(max_feature_entry)))
            # kv_rows.append((run_id, iteration, "Leaf {0} {1} min_feature_entry".format(k, dataset_type),
            #                 np.asscalar(min_feature_entry)))
            # kv_rows.append((run_id, iteration, "Leaf {0} {1} mean_feature_magnitude".format(k, dataset_type),
            #                 np.asscalar(mean_feature_magnitude)))
            # kv_rows.append((run_id, iteration, "Leaf {0} {1} max_feature_magnitude".format(k, dataset_type),
            #                 np.asscalar(max_feature_magnitude)))
            # kv_rows.append((run_id, iteration, "Leaf {0} {1} min_feature_magnitude".format(k, dataset_type),
            #                 np.asscalar(min_feature_magnitude)))
            # kv_rows.append((run_id, iteration, "Leaf {0} {1} std_feature_magnitude".format(k, dataset_type),
            #                 np.asscalar(std_feature_magnitude)))
        # Measure Information Gain
        total_info_gain = 0.0
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
            total_info_gain -= avg_info_gain
            kv_rows.append((run_id, iteration, "Dataset:{0} IG:{1}".format(dataset_type, k), avg_info_gain))
        kv_rows.append((run_id, iteration, "Dataset:{0} Total IG".format(dataset_type), total_info_gain))
        # Measure Branching Probabilities
        AccuracyCalculator.measure_branch_probs(run_id=run_id, iteration=iteration, dataset_type=dataset_type,
                                                branch_probs=branch_probs_dict, kv_rows=kv_rows)
        # Measure The Histogram of Branching Probabilities
        self.network.calculate_branch_probability_histograms(branch_probs=branch_probs_dict)
        # Measure Label Distribution
        self.label_distribution_analysis(run_id=run_id, iteration=iteration, kv_rows=kv_rows,
                                         leaf_true_labels_dict=leaf_true_labels_dict,
                                         dataset=dataset, dataset_type=dataset_type)
        # Measure Accuracy
        overall_count = 0.0
        overall_correct = 0.0
        confusion_matrix_db_rows = []
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            if node.index not in leaf_predicted_labels_dict:
                continue
            predicted = leaf_predicted_labels_dict[node.index]
            true_labels = leaf_true_labels_dict[node.index]
            if predicted.shape != true_labels.shape:
                raise Exception("Predicted and true labels counts do not hold.")
            correct_count = np.sum(predicted == true_labels)
            # Get the incorrect predictions by preparing a confusion matrix for each leaf
            sparse_confusion_matrix = {}
            for i in range(true_labels.shape[0]):
                true_label = true_labels[i]
                predicted_label = predicted[i]
                label_pair = (np.asscalar(true_label), np.asscalar(predicted_label))
                if label_pair not in sparse_confusion_matrix:
                    sparse_confusion_matrix[label_pair] = 0
                sparse_confusion_matrix[label_pair] += 1
            for k, v in sparse_confusion_matrix.items():
                confusion_matrix_db_rows.append((run_id, dataset_type.value, node.index, iteration, k[0], k[1], v))
            # Overall accuracy
            total_count = true_labels.shape[0]
            overall_correct += correct_count
            overall_count += total_count
            if total_count > 0:
                print("Leaf {0}: Sample Count:{1} Accuracy:{2}".format(node.index, total_count,
                                                                       float(correct_count) / float(total_count)))
            else:
                print("Leaf {0} is empty.".format(node.index))
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(overall_count, overall_correct / overall_count))
        total_accuracy = overall_correct / overall_count
        # Calculate modes
        self.network.modeTracker.calculate_modes(leaf_true_labels_dict=leaf_true_labels_dict,
                                                 dataset=dataset, dataset_type=dataset_type, kv_rows=kv_rows,
                                                 run_id=run_id, iteration=iteration)
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        return overall_correct / overall_count, confusion_matrix_db_rows

    def calculate_accuracy_after_compression(self, sess, run_id, iteration, dataset, dataset_type):
        kv_rows = []
        posterior_probs = {}
        branch_probs_dict = {}
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        final_features_dict = {}
        residue_posteriors_dict = {}
        # Run with mask off
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        while True:
            results, _ = self.network.eval_network(sess=sess, dataset=dataset, use_masking=False)
            if results is not None:
                for node in self.network.topologicalSortedNodes:
                    if not node.isLeaf:
                        # info_gain = results[self.network.get_variable_name(name="info_gain", node=node)]
                        branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs_dict, key=node.index,
                                                             array=branch_prob)
                        # if node.index not in info_gain_dict:
                        #     info_gain_dict[node.index] = []
                        # info_gain_dict[node.index].append(np.asscalar(info_gain))
                        continue
                    else:
                        posterior_prob = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                        # predicted_labels = np.argmax(posterior_prob, axis=1)
                        true_labels = results["Node{0}_label_tensor".format(node.index)]
                        final_features = results[self.network.get_variable_name(name="final_feature_final", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=posterior_probs, key=node.index,
                                                             array=posterior_prob)
                        # UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                        #                                      array=predicted_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                             array=true_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=final_features_dict, key=node.index,
                                                             array=final_features)
                        residue_posteriors = sess.run([self.network.evalDict["residue_probabilities"]],
                                                      feed_dict={self.network.residueInputTensor: final_features,
                                                                 self.network.classificationDropoutKeepProb: 1.0})
                        UtilityFuncs.concat_to_np_array_dict(dct=residue_posteriors_dict, key=node.index,
                                                             array=residue_posteriors[0])
            if dataset.isNewEpoch:
                break
        # Integrity check
        true_labels_list = list(leaf_true_labels_dict.values())
        for label_arr in true_labels_list:
            assert np.array_equal(label_arr, true_labels_list[0])
        # Collect branching statistics
        self.measure_branching_properties(sess=sess, run_id=run_id, iteration=iteration,
                                          dataset=dataset, dataset_type=dataset_type, kv_rows=kv_rows)
        # Method 1: Find the most confident leaf. If this leaf is indecisive, then find the most confident leaf.
        # If all confidents are indecisive, then pick the most confident leaf's prediction as a heuristic.
        root_node = self.network.nodes[0]
        sample_count = list(leaf_true_labels_dict.values())[0].shape[0]
        total_mode_prediction_count = 0
        total_correct_of_mode_predictions = 0
        samples_with_non_mode_predictions = set()
        wrong_samples_with_non_mode_predictions = set()
        true_labels_dict = {}
        leaf_nodes_dict = {}
        modes_per_leaves = self.network.modeTracker.get_modes()
        # Predict all samples which correspond to modes
        for sample_index in range(sample_count):
            curr_node = root_node
            probabilities_on_path = []
            while True:
                if not curr_node.isLeaf:
                    p_n_given_sample = branch_probs_dict[curr_node.index][sample_index, :]
                    child_nodes = self.network.dagObject.children(node=curr_node)
                    child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                    arg_max_index = np.asscalar(np.argmax(p_n_given_sample))
                    probabilities_on_path.append(p_n_given_sample[arg_max_index])
                    curr_node = child_nodes_sorted[arg_max_index]
                else:
                    sample_posterior = posterior_probs[curr_node.index][sample_index, :]
                    predicted_compressed_label = np.asscalar(np.argmax(sample_posterior))
                    true_label = leaf_true_labels_dict[curr_node.index][sample_index]
                    true_labels_dict[sample_index] = true_label
                    leaf_nodes_dict[sample_index] = curr_node.index
                    inverse_label_mapping = self.network.softmaxCompresser.inverseLabelMappings[curr_node.index]
                    predicted_label = inverse_label_mapping[predicted_compressed_label]
                    if predicted_label == -1:
                        samples_with_non_mode_predictions.add(sample_index)
                    else:
                        total_mode_prediction_count += 1
                        if predicted_label == true_label:
                            total_correct_of_mode_predictions += 1
                    break
        kv_rows.append((run_id, iteration, "{0} Mode Predictions".format(dataset_type), total_mode_prediction_count))
        kv_rows.append((run_id, iteration, "{0} Non Mode Predictions".format(dataset_type),
                        len(samples_with_non_mode_predictions)))
        kv_rows.append((run_id, iteration, "{0} Mode Correct Predictions".format(dataset_type),
                        total_correct_of_mode_predictions))
        kv_rows.append((run_id, iteration, "{0} Mode Accuracy".format(dataset_type),
                        float(total_correct_of_mode_predictions) /
                        float(total_mode_prediction_count)))
        # Handle all samples with non mode predictions Method 1
        # Try to correct non mode estimations with a simple heuristics:
        # 1) Check all leaves. Among the leaves which predicts the sample having a label within its modes, choose the
        # prediction with the highest confidence.
        # 2) If all leaves predict the sample as a non mode, pick the estimate with the highest confidence.
        method1_total_correct_non_mode_predictions = 0
        set_of_indecisive_samples = set()
        method1_total_correct_decisive_predictions = 0
        method1_total_correct_indecisive_predictions = 0
        for sample_index in samples_with_non_mode_predictions:
            curr_predicted_label = None
            curr_prediction_confidence = 0.0
            for node in self.network.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                sample_posterior = posterior_probs[node.index][sample_index, :]
                compressed_labels_sorted = np.argsort(sample_posterior)[::-1]
                inverse_label_mapping = self.network.softmaxCompresser.inverseLabelMappings[node.index]
                predicted_compressed_label0 = compressed_labels_sorted[0]
                predicted_label0 = inverse_label_mapping[predicted_compressed_label0]
                label0_probability = sample_posterior[predicted_compressed_label0]
                predicted_compressed_label1 = compressed_labels_sorted[1]
                predicted_label1 = inverse_label_mapping[predicted_compressed_label1]
                label1_probability = sample_posterior[predicted_compressed_label1]
                assert predicted_label0 != -1 or predicted_label1 != -1
                if predicted_label0 != -1:
                    if label0_probability + 1.0 > curr_prediction_confidence:
                        curr_prediction_confidence = label0_probability + 1.0
                        curr_predicted_label = predicted_label0
                else:
                    if label1_probability > curr_prediction_confidence:
                        curr_prediction_confidence = label1_probability
                        curr_predicted_label = predicted_label1
            if curr_prediction_confidence < 1.0:
                set_of_indecisive_samples.add(sample_index)
            if curr_predicted_label == true_labels_dict[sample_index]:
                method1_total_correct_non_mode_predictions += 1
                if curr_prediction_confidence < 1.0:
                    method1_total_correct_indecisive_predictions += 1
                else:
                    method1_total_correct_decisive_predictions += 1
        best_leaf_accuracy = (method1_total_correct_non_mode_predictions + total_correct_of_mode_predictions) / \
                             sample_count

        # Handle all samples with non mode predictions Method 2
        # Try to correct non mode estimations with the residue network:
        # If a sample is non-mode, then accept the residue network's inference about it as the final result.
        method2_total_correct_indecisive_predictions = 0
        for sample_index in set_of_indecisive_samples:
            residue_posterior = residue_posteriors_dict[leaf_nodes_dict[sample_index]][sample_index, :]
            residue_predicted_label = np.asscalar(np.argmax(residue_posterior))
            if residue_predicted_label == true_labels_dict[sample_index]:
                method2_total_correct_indecisive_predictions += 1
        method2_total_correct_non_mode_predictions = method2_total_correct_indecisive_predictions + \
                                                     method1_total_correct_decisive_predictions
        residue_correction_accuracy = (method2_total_correct_non_mode_predictions +
                                       total_correct_of_mode_predictions) / \
                                      sample_count

        kv_rows.append((run_id, iteration, "{0} Method 1 Correct Non Mode Predictions".format(dataset_type),
                        method1_total_correct_non_mode_predictions))
        kv_rows.append((run_id, iteration, "{0} Method 1 Non Mode Prediction Accuracy".format(dataset_type),
                        float(method1_total_correct_non_mode_predictions) /
                        float(len(samples_with_non_mode_predictions))))
        kv_rows.append((run_id, iteration, "{0} Method 2 Correct Non Mode Predictions".format(dataset_type),
                        method2_total_correct_non_mode_predictions))
        kv_rows.append((run_id, iteration, "{0} Method 2 Non Mode Prediction Accuracy".format(dataset_type),
                        float(method2_total_correct_non_mode_predictions) /
                        float(total_mode_prediction_count)))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        return best_leaf_accuracy, residue_correction_accuracy

    def measure_branching_properties(self, sess, run_id, iteration, dataset, dataset_type, kv_rows):
        branch_probs_dict_mask_on = {}
        info_gain_dict = {}
        leaf_true_labels_dict = {}
        leaf_predicted_labels_dict = {}
        # Run with mask on
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        while True:
            results, _ = self.network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            if results is not None:
                for node in self.network.topologicalSortedNodes:
                    if not node.isLeaf:
                        info_gain = results[self.network.get_variable_name(name="info_gain", node=node)]
                        branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs_dict_mask_on, key=node.index,
                                                             array=branch_prob)
                        if node.index not in info_gain_dict:
                            info_gain_dict[node.index] = []
                        info_gain_dict[node.index].append(np.asscalar(info_gain))
                    else:
                        posterior_prob = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                        predicted_labels = np.argmax(posterior_prob, axis=1)
                        true_labels = results["Node{0}_label_tensor".format(node.index)]
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                                                             array=predicted_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                             array=true_labels)
            if dataset.isNewEpoch:
                break
        # Measure Information Gain
        total_info_gain = 0.0
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
            total_info_gain -= avg_info_gain
            kv_rows.append((run_id, iteration, "Dataset:{0} IG:{1}".format(dataset_type, k), avg_info_gain))
        kv_rows.append((run_id, iteration, "Dataset:{0} Total IG".format(dataset_type), total_info_gain))
        # Measure Branching Probabilities
        AccuracyCalculator.measure_branch_probs(run_id=run_id, iteration=iteration, dataset_type=dataset_type,
                                                branch_probs=branch_probs_dict_mask_on, kv_rows=kv_rows)
        # Measure Label Distribution
        self.label_distribution_analysis(run_id=run_id, iteration=iteration, kv_rows=kv_rows,
                                         leaf_true_labels_dict=leaf_true_labels_dict,
                                         dataset=dataset, dataset_type=dataset_type)

    def calculate_accuracy_multipath(self, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_true_labels_dict = {}
        branch_activations = {}
        branch_probs = {}
        posterior_probs = {}
        hash_codes = {}
        while True:
            results, minibatch = self.network.eval_network(sess=sess, dataset=dataset, use_masking=False)
            if results is not None:
                for node in self.network.topologicalSortedNodes:
                    if not node.isLeaf:
                        branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                        activations = results[self.network.get_variable_name(name="activations", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs, key=node.index,
                                                             array=branch_prob)
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_activations, key=node.index,
                                                             array=activations)
                    else:
                        posterior_prob = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                        true_labels = results["Node{0}_label_tensor".format(node.index)]
                        UtilityFuncs.concat_to_np_array_dict(dct=posterior_probs, key=node.index,
                                                             array=posterior_prob)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                             array=true_labels)
                        if GlobalConstants.USE_SAMPLE_HASHING:
                            UtilityFuncs.concat_to_np_array_dict(dct=hash_codes, key=node.index,
                                                                 array=minibatch.hash_codes)

            if dataset.isNewEpoch:
                break
        sample_count = list(leaf_true_labels_dict.values())[0].shape[0]
        label_list = list(leaf_true_labels_dict.values())[0]
        for v in leaf_true_labels_dict.values():
            assert np.allclose(v, label_list)
        if GlobalConstants.USE_SAMPLE_HASHING and iteration >= 43201:
            hash_list = list(hash_codes.values())[0]
            for v in hash_codes.values():
                assert np.array_equal(v, hash_list)
            statistic_rows = self.prepare_sample_wise_statistics(run_id=run_id, iteration=iteration,
                                                                 hash_list=hash_list, sample_count=sample_count,
                                                                 branch_activations=branch_activations,
                                                                 branch_probs=branch_probs,
                                                                 posterior_probs=posterior_probs,
                                                                 true_labels=label_list)
            DbLogger.write_into_table(rows=statistic_rows, table=DbLogger.sample_wise_table, col_count=5)
        # total_correct = 0
        # total_mode_prediction_count = 0
        # total_correct_of_mode_predictions = 0
        # samples_with_non_mode_predictions = set()
        # wrong_samples_with_non_mode_predictions = set()
        # true_labels_dict = {}
        # modes_per_leaves = self.network.modeTracker.get_modes()
        threshold_dict = UtilityFuncs.distribute_evenly_to_threads(
            num_of_threads=GlobalConstants.SOFTMAX_DISTILLATION_CPU_COUNT,
            list_to_distribute=GlobalConstants.MULTIPATH_SCHEDULES)
        threads_dict = {}
        for thread_id in range(GlobalConstants.SOFTMAX_DISTILLATION_CPU_COUNT):
            threads_dict[thread_id] = MultipathCalculator(thread_id=thread_id, run_id=run_id, iteration=iteration,
                                                          threshold_list=threshold_dict[thread_id],
                                                          network=self.network,
                                                          sample_count=sample_count, label_list=label_list,
                                                          branch_probs=branch_probs, posterior_probs=posterior_probs)
            threads_dict[thread_id].start()
        all_results = []
        for thread in threads_dict.values():
            thread.join()
        for thread in threads_dict.values():
            all_results.extend(thread.kvRows)
        DbLogger.write_into_table(rows=all_results, table=DbLogger.multipath_results_table, col_count=6)

    def calculate_accuracy_with_route_correction(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        branch_probs = {}
        one_hot_branch_probs = {}
        posterior_probs = {}
        while True:
            results, _ = self.network.eval_network(sess=sess, dataset=dataset, use_masking=False)
            if results is not None:
                for node in self.network.topologicalSortedNodes:
                    if not node.isLeaf:
                        branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs, key=node.index,
                                                             array=branch_prob)
                    else:
                        posterior_prob = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                        true_labels = results["Node{0}_label_tensor".format(node.index)]
                        UtilityFuncs.concat_to_np_array_dict(dct=posterior_probs, key=node.index,
                                                             array=posterior_prob)
                        # UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                        #                                      array=predicted_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                             array=true_labels)
            if dataset.isNewEpoch:
                break
        label_dict = list(leaf_true_labels_dict.values())[0]
        for v in leaf_true_labels_dict.values():
            assert np.allclose(v, label_dict)
        root_node = self.network.nodes[0]
        # for node in self.topologicalSortedNodes:
        #     if not node.isLeaf:
        #         continue
        #     path = self.dagObject.get_shortest_path(source=root_node, dest=node)
        #     print("X")
        sample_count = list(leaf_true_labels_dict.values())[0].shape[0]
        total_correct = 0
        total_mode_prediction_count = 0
        total_correct_of_mode_predictions = 0
        samples_with_non_mode_predictions = set()
        wrong_samples_with_non_mode_predictions = set()
        true_labels_dict = {}
        modes_per_leaves = self.network.modeTracker.get_modes()
        # leaf_histograms = {}
        for sample_index in range(sample_count):
            curr_node = root_node
            probabilities_on_path = []
            while True:
                if not curr_node.isLeaf:
                    p_n_given_sample = branch_probs[curr_node.index][sample_index, :]
                    child_nodes = self.network.dagObject.children(node=curr_node)
                    child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                    arg_max_index = np.asscalar(np.argmax(p_n_given_sample))
                    probabilities_on_path.append(p_n_given_sample[arg_max_index])
                    curr_node = child_nodes_sorted[arg_max_index]
                else:
                    sample_posterior = posterior_probs[curr_node.index][sample_index, :]
                    predicted_label = np.asscalar(np.argmax(sample_posterior))
                    true_label = leaf_true_labels_dict[curr_node.index][sample_index]
                    # if curr_node.index not in leaf_histograms:
                    #     leaf_histograms[curr_node.index] = {}
                    # if true_label not in leaf_histograms[curr_node.index]:
                    #     leaf_histograms[curr_node.index][true_label] = 0
                    # leaf_histograms[curr_node.index][true_label] += 1
                    true_labels_dict[sample_index] = true_label
                    if predicted_label not in modes_per_leaves[curr_node.index]:
                        samples_with_non_mode_predictions.add(sample_index)
                        # This is just for analysis.
                        if true_label != predicted_label:
                            wrong_samples_with_non_mode_predictions.add(sample_index)
                    else:
                        total_mode_prediction_count += 1.0
                        if true_label == predicted_label:
                            total_correct_of_mode_predictions += 1.0
                            total_correct += 1.0
                    # if true_label == predicted_label:
                    #     total_correct += 1.0
                    # else:
                    #     if sample_index in samples_with_non_mode_predictions:
                    #         wrong_samples_with_non_mode_predictions.add(sample_index)
                    # else:
                    #     print("Wrong!")
                    break
        # Try to correct non mode estimations with a simple heuristics:
        # 1) Check all leaves. Among the leaves which predicts the sample having a label within its modes, choose the
        # prediction with the highest confidence.
        # 2) If all leaves predict the sample as a non mode, pick the estimate with the highest confidence.
        # First Method
        for sample_index in samples_with_non_mode_predictions:
            curr_predicted_label = None
            curr_prediction_confidence = 0.0
            for node in self.network.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                sample_posterior = posterior_probs[node.index][sample_index, :]
                leaf_modes = modes_per_leaves[node.index]
                predicted_label = np.asscalar(np.argmax(sample_posterior))
                prediction_confidence = sample_posterior[predicted_label] + float(predicted_label in leaf_modes)
                if prediction_confidence > curr_prediction_confidence:
                    curr_predicted_label = predicted_label
                    curr_prediction_confidence = prediction_confidence
            if curr_predicted_label == true_labels_dict[sample_index]:
                total_correct += 1
        corrected_accuracy = total_correct / sample_count
        print("Dataset:{0} Modified Accuracy={1}".format(dataset_type, corrected_accuracy))
        print("Total count of mode predictions={0}".format(total_mode_prediction_count))
        mode_prediction_accuracy = total_correct_of_mode_predictions / total_mode_prediction_count
        print("Mode prediction accuracy={0}".format(mode_prediction_accuracy))
        # Second Method
        avg_total_correct = 0
        for sample_index in samples_with_non_mode_predictions:
            selection_prob_dict = {root_node.index: 1.0}
            avg_posterior = None
            for node in self.network.topologicalSortedNodes:
                if not node.isLeaf:
                    p_n_given_sample = branch_probs[node.index][sample_index, :]
                    child_nodes = self.network.dagObject.children(node=node)
                    child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                    for child_index in range(len(child_nodes_sorted)):
                        selection_prob_dict[child_nodes_sorted[child_index].index] = \
                            selection_prob_dict[node.index] * p_n_given_sample[child_index]
                else:
                    sample_posterior = posterior_probs[node.index][sample_index, :]
                    if avg_posterior is None:
                        avg_posterior = selection_prob_dict[node.index] * sample_posterior
                    else:
                        avg_posterior = avg_posterior + (selection_prob_dict[node.index] * sample_posterior)
            avg_predicted_label = np.asscalar(np.argmax(avg_posterior))
            if avg_predicted_label == true_labels_dict[sample_index]:
                avg_total_correct += 1
        marginalized_corrected_accuracy = (total_correct_of_mode_predictions + avg_total_correct) / sample_count
        print("Marginalized prediction accuracy={0}".format(marginalized_corrected_accuracy))
        return corrected_accuracy, marginalized_corrected_accuracy

    def calculate_accuracy_with_residue_network(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_true_labels_dict = {}
        leaf_sample_indices_dict = {}
        leaf_posterior_probs_dict = {}
        residue_true_labels_dict = {}
        residue_sample_indices_dict = {}
        residue_posterior_probs_dict = {}
        modes_per_leaves = self.network.modeTracker.get_modes()
        while True:
            results, _ = self.network.eval_network(sess=sess, dataset=dataset, use_masking=False)
            if results is not None:
                for node in self.network.topologicalSortedNodes:
                    if not node.isLeaf:
                        branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_posterior_probs_dict, key=node.index,
                                                             array=branch_prob)
                    else:
                        posterior_prob = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                        true_labels = results["Node{0}_label_tensor".format(node.index)]
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_posterior_probs_dict, key=node.index,
                                                             array=posterior_prob)
                        # UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                        #                                      array=predicted_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                             array=true_labels)
                residue_posterior_prob = results["residue_probabilities"]
                residue_true_labels = results["residue_labels"]
                residue_sample_indices = results["residue_indices"]
                UtilityFuncs.concat_to_np_array_dict(dct=residue_posterior_probs_dict, key=-1,
                                                     array=residue_posterior_prob)
                UtilityFuncs.concat_to_np_array_dict(dct=residue_true_labels_dict, key=-1,
                                                     array=residue_true_labels)
                UtilityFuncs.concat_to_np_array_dict(dct=residue_sample_indices_dict, key=-1,
                                                     array=residue_sample_indices)
            if dataset.isNewEpoch:
                break
        label_dict = list(leaf_true_labels_dict.values())[0]
        residue_true_labels = residue_true_labels_dict[-1]
        residue_posterior_prob = residue_posterior_probs_dict[-1]
        residue_sample_indices = residue_sample_indices_dict[-1]
        for v in leaf_true_labels_dict.values():
            assert np.allclose(v, label_dict)
        assert np.allclose(label_dict, residue_true_labels)
        sample_count = list(leaf_true_labels_dict.values())[0].shape[0]
        root_node = self.network.nodes[0]
        # Accuracy measurements
        mode_samples_count = 0
        non_mode_samples_count = 0
        mode_correct_count = 0
        non_mode_correct_count = 0
        mode_accuracy = 0
        non_mode_accuracy = 0
        residue_mode_samples_count = 0
        residue_non_mode_samples_count = 0
        residue_mode_correct_count = 0
        residue_non_mode_correct_count = 0
        true_labels_dict = {}
        samples_with_mode_predictions = set()
        samples_with_non_mode_predictions = set()
        # Accuracy on tree
        for sample_index in range(sample_count):
            curr_node = root_node
            probabilities_on_path = []
            while True:
                if not curr_node.isLeaf:
                    p_n_given_sample = leaf_posterior_probs_dict[curr_node.index][sample_index, :]
                    child_nodes = self.network.dagObject.children(node=curr_node)
                    child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
                    arg_max_index = np.asscalar(np.argmax(p_n_given_sample))
                    probabilities_on_path.append(p_n_given_sample[arg_max_index])
                    curr_node = child_nodes_sorted[arg_max_index]
                else:
                    sample_posterior = leaf_posterior_probs_dict[curr_node.index][sample_index, :]
                    predicted_label = np.asscalar(np.argmax(sample_posterior))
                    true_label = leaf_true_labels_dict[curr_node.index][sample_index]
                    true_labels_dict[sample_index] = true_label
                    if predicted_label not in modes_per_leaves[curr_node.index]:
                        samples_with_non_mode_predictions.add(sample_index)
                        non_mode_samples_count += 1.0
                        if true_label == predicted_label:
                            non_mode_correct_count += 1.0
                    else:
                        samples_with_mode_predictions.add(sample_index)
                        mode_samples_count += 1.0
                        if true_label == predicted_label:
                            mode_correct_count += 1.0
                    break
        # Accuracy on residue network
        for sample_index in range(sample_count):
            residue_posterior = residue_posterior_prob[sample_index, :]
            predicted_label = np.asscalar(np.argmax(residue_posterior))
            true_label = residue_true_labels[sample_index]
            if sample_index in samples_with_non_mode_predictions:
                residue_non_mode_samples_count += 1.0
                if true_label == predicted_label:
                    residue_non_mode_correct_count += 1.0
            else:
                residue_mode_samples_count += 1.0
                if true_label == predicted_label:
                    residue_mode_correct_count += 1.0
        print("Number of mode predictions:{0}".format(mode_samples_count))
        print("Number of non mode predictions:{0}".format(non_mode_samples_count))
        print("Mode predictions accuracy:{0}".format(mode_correct_count / mode_samples_count))
        print("Non mode predictions accuracy:{0}".format(non_mode_correct_count / non_mode_samples_count))
        print("Total Accuracy:{0}".format((mode_correct_count + non_mode_correct_count) / (mode_samples_count +
                                                                                           non_mode_samples_count)))
        print("Residue mode accuracy:{0}".format(residue_mode_correct_count / residue_mode_samples_count))
        print(
            "Residue non mode accuracy:{0}".format(residue_non_mode_correct_count / residue_non_mode_samples_count))
        print("Residue Total accuracy:{0}".format((residue_mode_correct_count + residue_non_mode_correct_count) /
                                                  (residue_mode_samples_count + residue_non_mode_samples_count)))
