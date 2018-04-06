import numpy as np

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class AccuracyCalculator:
    def __init__(self, network):
        self.network = network
        self.modesHistory = []

    def calculate_accuracy(self, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        branch_probs = {}
        while True:
            results = self.network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            batch_sample_count = 0.0
            for node in self.network.topologicalSortedNodes:
                if not node.isLeaf:
                    info_gain = results[self.network.get_variable_name(name="info_gain", node=node)]
                    branch_prob = results[self.network.get_variable_name(name="p(n|x)", node=node)]
                    UtilityFuncs.concat_to_np_array_dict(dct=branch_probs, key=node.index, array=branch_prob)
                    if node.index not in info_gain_dict:
                        info_gain_dict[node.index] = []
                    info_gain_dict[node.index].append(np.asscalar(info_gain))
                    continue
                if results[self.network.get_variable_name(name="is_open", node=node)] == 0.0:
                    continue
                posterior_probs = results[self.network.get_variable_name(name="posterior_probs", node=node)]
                true_labels = results["Node{0}_label_tensor".format(node.index)]
                # batch_sample_count += results[self.get_variable_name(name="sample_count", node=node)]
                predicted_labels = np.argmax(posterior_probs, axis=1)
                batch_sample_count += predicted_labels.shape[0]
                UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                                                     array=predicted_labels)
                UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                     array=true_labels)
            if batch_sample_count != GlobalConstants.EVAL_BATCH_SIZE:
                raise Exception("Incorrect batch size:{0}".format(batch_sample_count))
            if dataset.isNewEpoch:
                break
        print("****************Dataset:{0}****************".format(dataset_type))
        # Measure Information Gain
        total_info_gain = 0.0
        kv_rows = []
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
            total_info_gain -= avg_info_gain
            kv_rows.append((run_id, iteration, "Dataset:{0} IG:{1}".format(dataset_type, k), avg_info_gain))
        kv_rows.append((run_id, iteration, "Dataset:{0} Total IG".format(dataset_type), total_info_gain))
        # Measure Branching Probabilities
        for k, v in branch_probs.items():
            p_n = np.mean(v, axis=0)
            print("p_{0}(n)={1}".format(k, p_n))
        # Measure The Histogram of Branching Probabilities
        self.network.calculate_branch_probability_histograms(branch_probs=branch_probs)
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

    def calculate_accuracy_with_route_correction(self, sess, dataset, dataset_type):
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        info_gain_dict = {}
        branch_probs = {}
        one_hot_branch_probs = {}
        posterior_probs = {}
        while True:
            results = self.network.eval_network(sess=sess, dataset=dataset, use_masking=False)
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
        dataset.set_current_data_set_type(dataset_type=dataset_type)
        leaf_true_labels_dict = {}
        leaf_sample_indices_dict = {}
        leaf_posterior_probs_dict = {}
        residue_true_labels_dict = {}
        residue_sample_indices_dict = {}
        residue_posterior_probs_dict = {}
        modes_per_leaves = self.network.modeTracker.get_modes()
        while True:
            results = self.network.eval_network(sess=sess, dataset=dataset, use_masking=False)
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