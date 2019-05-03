import tensorflow as tf
import numpy as np

from algorithms.accuracy_calculator import AccuracyCalculator
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle import Jungle
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


class JungleNoStitch(Jungle):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                         dataset)
        # self.unitTestList = [self.test_stitching]

    def apply_decision(self, node, branching_feature):
        assert node.nodeType == NodeType.h_node
        node.H_output = branching_feature
        node_degree = self.degreeList[node.depth + 1]
        if node_degree > 1:
            # Step 1: Create Hyperplanes
            ig_feature_size = node.H_output.get_shape().as_list()[-1]
            hyperplane_weights = tf.Variable(
                tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
            hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
            if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
                node.H_output = tf.layers.batch_normalization(inputs=node.H_output,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain, tf.bool))
            # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
            activations = tf.matmul(node.H_output, hyperplane_weights) + hyperplane_biases
            node.activationsDict[node.index] = activations
            decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
            p_F_given_x = tf.nn.softmax(decayed_activation)
            p_c_given_x = self.oneHotLabelTensor
            node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                      balance_coefficient=self.informationGainBalancingCoefficient)
            # Step 3:
            # If training: Sample Z from p(F|x) using Gumbel-Max trick
            # If testing: Pick Z = argmax_F p(F|x)
            category_count = tf.constant(node_degree)
            sampled_indices = self.sample_from_categorical(probs=p_F_given_x, batch_size=self.batchSize,
                                                           category_count=category_count)
            arg_max_indices = tf.argmax(p_F_given_x, axis=1, output_type=tf.int32)
            sampled_one_hot_matrix = tf.one_hot(sampled_indices, category_count)
            arg_max_one_hot_matrix = tf.one_hot(arg_max_indices, category_count)
            node.conditionProbabilities = tf.where(self.isTrain > 0, sampled_one_hot_matrix, arg_max_one_hot_matrix)
            # Reporting
            node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
            node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
            node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
            node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
            node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
            node.evalDict[UtilityFuncs.get_variable_name(name="p(n|x)", node=node)] = p_F_given_x
            node.evalDict[
                UtilityFuncs.get_variable_name(name="sampled_indices", node=node)] = sampled_indices
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_indices", node=node)] = arg_max_indices
            node.evalDict[
                UtilityFuncs.get_variable_name(name="sampled_one_hot_matrix", node=node)] = sampled_one_hot_matrix
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_one_hot_matrix", node=node)] = arg_max_one_hot_matrix
        else:
            node.conditionProbabilities = tf.ones_like(tensor=self.labelTensor, dtype=tf.float32)
        node.evalDict[
            UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = node.conditionProbabilities
        node.F_output = node.F_input

    def stitch_samples(self, node):
        assert node.nodeType == NodeType.h_node
        parents = self.dagObject.parents(node=node)
        # Layer 0 h_node. This receives non-partitioned, complete minibatch from the root node. No stitching needed.
        if len(parents) == 1:
            assert parents[0].nodeType == NodeType.root_node and node.depth == 0
            node.F_input = parents[0].F_output
            node.H_input = None
        # Need stitching
        else:
            # Get all F nodes in the same layer
            parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=node)
                              if f_node.nodeType == NodeType.f_node]
            parent_h_nodes = [h_node for h_node in self.dagObject.parents(node=node)
                              if h_node.nodeType == NodeType.h_node]
            assert len(parent_h_nodes) == 1
            parent_h_node = parent_h_nodes[0]
            parent_f_nodes = sorted(parent_f_nodes, key=lambda f_node: f_node.index)
            assert all([f_node.H_output is None for f_node in parent_f_nodes])
            f_inputs = [node.F_output for node in parent_f_nodes]
            # Get condition probabilities
            dependencies = []
            dependencies.extend(f_inputs)
            dependencies.append(parent_h_node.conditionProbabilities)
            with tf.control_dependencies(dependencies):
                f_weighted_list = []
                for f_index, f_input in enumerate(f_inputs):
                    f_weighted_list.append(
                        JungleNoStitch.multiply_tensor_with_branch_weights(
                            weights=parent_h_node.conditionProbabilities[:, f_index],
                            tensor=f_input))
                node.F_input = tf.add_n(f_weighted_list)
                node.H_input = parent_h_node.H_output
        node.evalDict[UtilityFuncs.get_variable_name(name="F_input", node=node)] = node.F_input

    @staticmethod
    def multiply_tensor_with_branch_weights(weights, tensor):
        _w = tf.identity(weights)
        input_dim = len(tensor.get_shape().as_list())
        assert input_dim == 2 or input_dim == 4
        for _ in range(input_dim - 1):
            _w = tf.expand_dims(_w, axis=-1)
        weighted_tensor = tf.multiply(tensor, _w)
        return weighted_tensor

    def mask_input_nodes(self, node):
        node.labelTensor = self.labelTensor
        if node.nodeType == NodeType.root_node:
            node.F_input = self.dataTensor
            node.H_input = None
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.conditionProbabilities = tf.ones_like(tensor=self.labelTensor, dtype=tf.float32)
            # For reporting
            node.sampleCountTensor = tf.reduce_sum(node.conditionProbabilities)
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = node.sampleCountTensor
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            node.evalDict[UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = \
                node.conditionProbabilities
        elif node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node:
            # raise NotImplementedError()
            parents = self.dagObject.parents(node=node)
            assert len(parents) == 1 and parents[0].nodeType == NodeType.h_node
            parent_node = parents[0]
            sibling_order_index = self.get_node_sibling_index(node=node)
            with tf.control_dependencies([parent_node.F_output,
                                          parent_node.H_output,
                                          parent_node.conditionProbabilities]):
                node.F_input = tf.identity(parent_node.F_output)
                node.H_input = tf.identity(parent_node.H_output)
                if len(parent_node.conditionProbabilities.get_shape().as_list()) > 1:
                    node.conditionProbabilities = tf.identity(parent_node.conditionProbabilities[:,
                                                              sibling_order_index])
                else:
                    node.conditionProbabilities = tf.identity(parent_node.conditionProbabilities)
                # For reporting
                node.sampleCountTensor = tf.reduce_sum(node.conditionProbabilities)
                is_used = tf.cast(node.sampleCountTensor, tf.float32) > 0.0
                node.isOpenIndicatorTensor = tf.where(is_used, 1.0, 0.0)
                # node.conditionIndices = tf.identity(parent_node.conditionIndices[sibling_order_index])
                node.evalDict[self.get_variable_name(name="sample_count", node=node)] = node.sampleCountTensor
                node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = \
                    node.conditionProbabilities

    def update_params_with_momentum(self, sess, dataset, epoch, iteration):
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.BATCH_SIZE
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.CURR_BATCH_SIZE)
        if minibatch is None:
            return None, None, None
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                           is_train=True, use_masking=True)
        # Prepare result tensors to collect
        run_ops = self.get_run_ops()
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        # print("Before Update Iteration:{0}".format(iteration))
        results = sess.run(run_ops, feed_dict=feed_dict)
        # print("After Update Iteration:{0}".format(iteration))
        lr = results[1]
        sample_counts = results[2]
        is_open_indicators = results[3]
        # Unit Tests
        if GlobalConstants.USE_UNIT_TESTS:
            for test in self.unitTestList:
                test(results[-1])
        return lr, sample_counts, is_open_indicators

    def calculate_accuracy(self, calculation_type, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        final_features_dict = {}
        info_gain_dict = {}
        branch_probs_dict = {}
        arg_max_indices_dict = {}
        while True:
            results, _ = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            if results is not None:
                batch_sample_count = 0.0
                for node in self.topologicalSortedNodes:
                    if node.nodeType == NodeType.h_node:
                        node_degree = self.degreeList[node.depth + 1]
                        if node_degree == 1:
                            continue
                        info_gain = results[self.get_variable_name(name="info_gain", node=node)]
                        branch_prob = results[self.get_variable_name(name="p(n|x)", node=node)]
                        arg_max_indices = results[UtilityFuncs.get_variable_name(name="arg_max_indices", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs_dict, key=node.index, array=branch_prob)
                        UtilityFuncs.concat_to_np_array_dict(dct=arg_max_indices_dict, key=node.index,
                                                             array=arg_max_indices)
                        if node.index not in info_gain_dict:
                            info_gain_dict[node.index] = []
                        info_gain_dict[node.index].append(np.asscalar(info_gain))
                        continue
                    elif node.nodeType == NodeType.leaf_node:
                        posterior_probs = results[self.get_variable_name(name="posterior_probs", node=node)]
                        true_labels = results[UtilityFuncs.get_variable_name(name="labelTensor", node=node)]
                        final_features = results[self.get_variable_name(name="final_feature_final", node=node)]
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
        print("****************Dataset:{0}****************".format(dataset_type))
        kv_rows = []
        # Measure Information Gain
        total_info_gain = 0.0
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
            total_info_gain -= avg_info_gain
            kv_rows.append((run_id, iteration, "Dataset:{0} IG:{1}".format(dataset_type, k), avg_info_gain))
        kv_rows.append((run_id, iteration, "Dataset:{0} Total IG".format(dataset_type), total_info_gain))
        # Measure h node label distribution
        assert len(leaf_true_labels_dict) == 1
        self.measure_h_node_label_distribution(arg_max_dict=arg_max_indices_dict,
                                               labels_arr=list(leaf_true_labels_dict.values())[0])
        # Measure Branching Probabilities
        AccuracyCalculator.measure_branch_probs(run_id=run_id, iteration=iteration, dataset_type=dataset_type,
                                                branch_probs=branch_probs_dict, kv_rows=kv_rows)
        # Measure The Histogram of Branching Probabilities
        self.calculate_branch_probability_histograms(branch_probs=branch_probs_dict)
        # Measure Label Distribution
        self.label_distribution_analysis(run_id=run_id, iteration=iteration, kv_rows=kv_rows,
                                         leaf_true_labels_dict=leaf_true_labels_dict,
                                         dataset=dataset, dataset_type=dataset_type)
        # # Measure Accuracy
        overall_count = 0.0
        overall_correct = 0.0
        confusion_matrix_db_rows = []
        for node in self.topologicalSortedNodes:
            if not node.nodeType == NodeType.leaf_node:
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
        # self.network.modeTracker.calculate_modes(leaf_true_labels_dict=leaf_true_labels_dict,
        #                                          dataset=dataset, dataset_type=dataset_type, kv_rows=kv_rows,
        #                                          run_id=run_id, iteration=iteration)
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        return overall_correct / overall_count, confusion_matrix_db_rows

    def measure_h_node_label_distribution(self, arg_max_dict, labels_arr):
        for k, v in arg_max_dict.items():
            print("X")

    # Unit test methods
    def test_stitching(self, eval_dict):
        h_nodes = [node for node in self.topologicalSortedNodes if node.nodeType == NodeType.h_node]
        for h_node in h_nodes:
            if len(self.dagObject.parents(node=h_node)) == 1:
                continue
            parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=h_node)
                              if f_node.nodeType == NodeType.f_node]
            parent_f_nodes = sorted(parent_f_nodes, key=lambda f_node: f_node.index)
            parent_h_nodes = [h_node for h_node in self.dagObject.parents(node=h_node)
                              if h_node.nodeType == NodeType.h_node]
            assert len(parent_h_nodes) == 1
            f_input = eval_dict[UtilityFuncs.get_variable_name(name="F_input", node=h_node)]
            condition_probabilities = eval_dict[UtilityFuncs.get_variable_name(name="conditionProbabilities",
                                                                               node=parent_h_nodes[0])]
            f_outputs_prev_layer = [eval_dict[UtilityFuncs.get_variable_name(name="F_output", node=node)]
                                    for node in parent_f_nodes]
            f_input_manual = np.zeros_like(f_input)
            for r in range(condition_probabilities.shape[0]):
                selected_node_idx = np.asscalar(np.argmax(condition_probabilities[r, :]))
                f_input_manual[r, :] = f_outputs_prev_layer[selected_node_idx][r, :]
            assert np.allclose(f_input, f_input_manual)
