from algorithms.cross_validation import CrossValidation
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import tensorflow as tf
import itertools
from random import shuffle

from simple_tf.global_params import GlobalConstants


class NetworkOutputs:
    def __init__(self):
        self.featureVectorsDict = {}
        self.logitsDict = {}
        self.posteriorsDict = {}
        self.oneHotLabelsDict = {}


class SoftmaxCompresser:
    def __init__(self):
        pass

    @staticmethod
    def compress_network_softmax(network, sess, dataset, run_id):
        # Get all final feature vectors for all leaves, for the complete training set.
        softmax_weights = {}
        softmax_biases = {}
        network_outputs = {}
        for dataset_type in [DatasetTypes.training, DatasetTypes.test]:
            network_output = NetworkOutputs()
            dataset.set_current_data_set_type(dataset_type=dataset_type)
            while True:
                results = network.eval_network(sess=sess, dataset=dataset, use_masking=True)
                for node in network.topologicalSortedNodes:
                    if not node.isLeaf:
                        continue
                    leaf_node = node
                    posterior_ref = network.get_variable_name(name="posterior_probs", node=leaf_node)
                    posterior_probs = results[posterior_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.posteriorsDict, key=leaf_node.index,
                                                         array=posterior_probs)
                    final_feature_ref = network.get_variable_name(name="final_eval_feature", node=leaf_node)
                    final_leaf_features = results[final_feature_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.featureVectorsDict, key=leaf_node.index,
                                                         array=final_leaf_features)
                    one_hot_label_ref = "Node{0}_one_hot_label_tensor".format(leaf_node.index)
                    one_hot_labels = results[one_hot_label_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.oneHotLabelsDict, key=leaf_node.index,
                                                         array=one_hot_labels)
                    logits_ref = network.get_variable_name(name="logits", node=leaf_node)
                    logits = results[logits_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.logitsDict, key=leaf_node.index,
                                                         array=logits)
                    softmax_weights_ref = network.get_variable_name(name="fc_softmax_weights", node=leaf_node)
                    softmax_weights[leaf_node.index] = results[softmax_weights_ref]
                    softmax_biases_ref = network.get_variable_name(name="fc_softmax_biases", node=leaf_node)
                    softmax_biases[leaf_node.index] = results[softmax_biases_ref]
                if dataset.isNewEpoch:
                    network_outputs[dataset_type] = network_output
                    break
        # Train all leaf classifiers by distillation
        training_data = network_outputs[DatasetTypes.training]
        for node in network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            leaf_node = node
            softmax_weight = softmax_weights[leaf_node.index]
            softmax_bias = softmax_biases[leaf_node.index]
            feature_vectors = training_data.featureVectorsDict[leaf_node.index]
            logits = training_data.logitsDict[leaf_node.index]
            probs = training_data.posteriorsDict[leaf_node.index]
            one_hot_labels = training_data.oneHotLabelsDict[leaf_node.index]
            # Unit Test
            SoftmaxCompresser.assert_prob_correctness(softmax_weights=softmax_weight, softmax_biases=softmax_bias,
                                                      features=feature_vectors, logits=logits, probs=probs)
            # Train compresser
            SoftmaxCompresser.train_distillation_network(sess=sess, network=network, leaf_node=leaf_node,
                                                         training_data=network_outputs[DatasetTypes.training],
                                                         test_data=network_outputs[DatasetTypes.test], run_id=run_id)

            # # Create compressed probabilities
            # SoftmaxCompresser.build_compressed_probabilities(network=network, leaf_node=leaf_node, posteriors=probs,
            #                                                  one_hot_labels=one_hot_labels)
            # # Create symbolic network for distillation
        print("X")

    @staticmethod
    def assert_prob_correctness(softmax_weights, softmax_biases, features, logits, probs):
        logits_np = np.dot(features, softmax_weights) + softmax_biases
        exp_logits = np.exp(logits_np)
        logit_sums = np.sum(exp_logits, 1).reshape(exp_logits.shape[0], 1)
        manual_probs1 = exp_logits / logit_sums

        exp_logits2 = np.exp(logits)
        logit_sums2 = np.sum(exp_logits2, 1).reshape(exp_logits2.shape[0], 1)
        manual_probs2 = exp_logits2 / logit_sums2

        is_equal1 = np.allclose(probs, manual_probs1)
        print("is_equal1={0}".format(is_equal1))

        is_equal2 = np.allclose(probs, manual_probs2)
        print("is_equal2={0}".format(is_equal2))

        assert is_equal1
        assert is_equal2

    @staticmethod
    def train_distillation_network(sess, network, leaf_node, training_data, test_data, run_id):
        training_logits = training_data.logitsDict[leaf_node.index]
        training_one_hot_labels = training_data.oneHotLabelsDict[leaf_node.index]
        training_features = training_data.featureVectorsDict[leaf_node.index]
        test_logits = test_data.logitsDict[leaf_node.index]
        test_one_hot_labels = test_data.oneHotLabelsDict[leaf_node.index]
        test_features = test_data.featureVectorsDict[leaf_node.index]
        assert training_logits.shape[0] == training_one_hot_labels.shape[0]
        assert training_logits.shape[0] == training_features.shape[0]
        logit_dim = training_logits.shape[1]
        features_dim = training_features.shape[1]
        modes_per_leaves = network.modeTracker.get_modes()
        compressed_class_count = len(modes_per_leaves[leaf_node.index]) + 1
        # p: The tempered posteriors, which have been squashed.
        p = tf.placeholder(tf.float32, shape=(None, compressed_class_count))
        # t: The squashed one hot labels
        t = tf.placeholder(tf.float32, shape=(None, compressed_class_count))
        features_tensor = tf.placeholder(tf.float32, shape=(None, features_dim))
        soft_labels_cost_weight = tf.placeholder(tf.float32)
        hard_labels_cost_weight = tf.placeholder(tf.float32)
        l2_loss_weight = tf.placeholder(tf.float32)
        # Get new class count: Mode labels + Outliers. Init the new classifier hyperplanes.
        softmax_weights = tf.Variable(
            tf.truncated_normal([features_dim, compressed_class_count],
                                stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="distilled_fc_softmax_weights", node=leaf_node))
        softmax_biases = tf.Variable(
            tf.constant(0.1, shape=[compressed_class_count], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="distilled_fc_softmax_biases", node=leaf_node))
        # Compressed softmax probabilities
        compressed_logits = tf.matmul(features_tensor, softmax_weights) + softmax_biases
        # Prepare the loss function, according to Hinton's Distillation Recipe
        # Term 1: Cross entropy between the tempered, squashed posteriors p and q: H(p,q)
        soft_loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=compressed_logits)
        soft_loss = soft_labels_cost_weight * tf.reduce_mean(soft_loss_vec)
        # Term 2: Cross entropy between the hard labels and q
        hard_loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=compressed_logits)
        hard_loss = hard_labels_cost_weight * tf.reduce_mean(hard_loss_vec)
        # Term 3: L2 loss for softmax weights
        weight_l2 = l2_loss_weight * tf.nn.l2_loss(softmax_weights)
        # Total loss
        distillation_loss = soft_loss + hard_loss + weight_l2
        # Softmax Output
        compressed_softmax_output = tf.nn.softmax(compressed_logits)
        # Gradients (For debug purposes)
        grad_soft_loss = tf.gradients(ys=soft_loss, xs=[softmax_weights, softmax_biases])
        grad_hard_loss = tf.gradients(ys=hard_loss, xs=[softmax_weights, softmax_biases])
        grad_sm_weights = tf.gradients(ys=weight_l2, xs=[softmax_weights])
        # Train by cross-validation
        temperature_list = [1.0]
        soft_loss_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        hard_loss_weights = [1.0]
        l2_weights = [0.0]
        learning_rates = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
                          0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        cross_validation_repeat_count = 10
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[temperature_list, soft_loss_weights,
                                                                              hard_loss_weights,
                                                                              l2_weights, learning_rates])
        duplicate_cartesians = []
        for tpl in cartesian_product:
            duplicate_cartesians.extend(list(itertools.repeat(tpl, cross_validation_repeat_count)))
        results_dict = {}

        #         kv_rows.append((run_id, iteration, "Leaf {0} Modes".format(node.index), mode_txt))
        #     print("Node{0} Label Distribution: {1}".format(node.index, distribution_str))
        # # if dataset_type == DatasetTypes.training and total_mode_count != GlobalConstants.NUM_LABELS:
        # #     raise Exception("total_mode_count != GlobalConstants.NUM_LABELS")
        # # Measure overall information gain
        # if dataset_type == DatasetTypes.training:
        #     kv_rows.append((run_id, iteration, "Total Mode Count", total_mode_count))

        # A new run for each tuple
        for tpl in duplicate_cartesians:
            temperature = tpl[0]
            soft_loss_weight = tpl[1]
            hard_loss_weight = tpl[2]
            l2_weight = tpl[3]
            lr = tpl[4]
            kv_rows = []
            # Build the optimizer
            global_step = tf.Variable(name="global_step", initial_value=0, trainable=False)
            learning_rate = tf.train.exponential_decay(lr, global_step,
                                                       GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT,
                                                       GlobalConstants.SOFTMAX_DISTILLATION_DECAY, staircase=True)
            trainer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(distillation_loss,
                                                                              global_step=global_step)
            # Init variables
            all_variables = tf.global_variables()
            vars_to_init = [var for var in all_variables if "/Momentum" in var.name]
            vars_to_init.extend([softmax_weights, softmax_biases, global_step])
            # Init variables
            init_op = tf.variables_initializer(vars_to_init)
            # Build the tempered posteriors
            training_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=training_logits,
                                                                                        temperature=temperature)
            test_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=test_logits,
                                                                                    temperature=temperature)
            # Get the compressed probabilities
            training_compressed_posteriors, training_compressed_one_hot_entries = \
                SoftmaxCompresser.build_compressed_probabilities(network=network, leaf_node=leaf_node,
                                                                 posteriors=training_tempered_posteriors,
                                                                 one_hot_labels=training_one_hot_labels)
            test_compressed_posteriors, test_compressed_one_hot_entries = \
                SoftmaxCompresser.build_compressed_probabilities(network=network, leaf_node=leaf_node,
                                                                 posteriors=test_tempered_posteriors,
                                                                 one_hot_labels=test_one_hot_labels)

            # Training sets
            training_sample_count = training_features.shape[0]
            training_indices = list(range(training_sample_count))
            shuffle(training_indices)
            random_indices = np.random.uniform(0, training_sample_count,
                                               GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE).astype(int).tolist()
            training_p = training_compressed_posteriors[training_indices]
            training_t = training_compressed_one_hot_entries[training_indices]
            training_x = training_features[training_indices]
            training_indices.extend(random_indices)
            training_p_wrapped = training_compressed_posteriors[training_indices]
            training_t_wrapped = training_compressed_one_hot_entries[training_indices]
            training_x_wrapped = training_features[training_indices]
            # Test sets
            test_sample_count = test_features.shape[0]
            test_indices = list(range(test_sample_count))
            test_p = test_compressed_posteriors[test_indices]
            test_t = test_compressed_one_hot_entries[test_indices]
            test_x = test_features[test_indices]
            # Calculate accuracy on the training set
            training_accuracy_full = \
                SoftmaxCompresser.calculate_compressed_accuracy(posteriors=training_p, one_hot_labels=training_t)
            # Calculate accuracy on the validation set
            test_accuracy_full = \
                SoftmaxCompresser.calculate_compressed_accuracy(posteriors=test_p, one_hot_labels=test_t)
            kv_rows.append((run_id, -1,
                            "Leaf:{0} Training Accuracy Full".format(leaf_node.index), training_accuracy_full))
            kv_rows.append((run_id, -1,
                            "Leaf:{0} Test Accuracy Full".format(leaf_node.index), test_accuracy_full))
            # Train
            batch_size = int(float(training_sample_count) * GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO)
            # GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE
            # Init softmax parameters
            sess.run(init_op)
            iteration = 0
            for epoch_id in range(GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT):
                curr_index = 0
                while True:
                    p_batch = training_p_wrapped[curr_index:curr_index + batch_size]
                    t_batch = training_t_wrapped[curr_index:curr_index + batch_size]
                    features_batch = training_x_wrapped[curr_index:curr_index + batch_size]
                    feed_dict = {p: p_batch, t: t_batch, features_tensor: features_batch,
                                 soft_labels_cost_weight: soft_loss_weight,
                                 hard_labels_cost_weight: hard_loss_weight,
                                 l2_loss_weight: l2_weight}
                    run_ops = [grad_soft_loss,
                               grad_hard_loss,
                               grad_sm_weights,
                               trainer,
                               learning_rate]
                    results = sess.run(run_ops, feed_dict=feed_dict)
                    iteration += 1
                    print("Iteration:{0} Learning Rate:{1}".format(iteration, results[-1]))
                    grad_soft_loss_weight_mag = np.linalg.norm(results[0][0])
                    grad_soft_loss_bias_mag = np.linalg.norm(results[0][1])
                    grad_hard_loss_weight_mag = np.linalg.norm(results[1][0])
                    grad_hard_loss_bias_mag = np.linalg.norm(results[1][1])
                    print("grad_soft_loss_weight_mag={0}".format(grad_soft_loss_weight_mag))
                    print("grad_soft_loss_bias_mag={0}".format(grad_soft_loss_bias_mag))
                    print("grad_hard_loss_weight_mag={0}".format(grad_hard_loss_weight_mag))
                    print("grad_hard_loss_bias_mag={0}".format(grad_hard_loss_bias_mag))
                    curr_index += batch_size
                    if curr_index >= training_sample_count:
                        # Evaluate on training set
                        training_results = sess.run([compressed_softmax_output, distillation_loss],
                                                    feed_dict={p: training_p,
                                                               t: training_t,
                                                               features_tensor: training_x,
                                                               soft_labels_cost_weight: soft_loss_weight,
                                                               hard_labels_cost_weight: hard_loss_weight,
                                                               l2_loss_weight: l2_weight})
                        training_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                            posteriors=training_results[0], one_hot_labels=training_t)
                        # Evaluate on test set
                        test_results = sess.run([compressed_softmax_output, distillation_loss],
                                                feed_dict={p: test_p,
                                                           t: test_t,
                                                           features_tensor: test_x,
                                                           soft_labels_cost_weight: soft_loss_weight,
                                                           hard_labels_cost_weight: hard_loss_weight,
                                                           l2_loss_weight: l2_weight})
                        test_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                            posteriors=test_results[0], one_hot_labels=test_t)
                        print("Uncompressed Training Accuracy:{0}".format(training_accuracy_full))
                        print("Uncompressed Test Accuracy:{0}".format(test_accuracy_full))
                        print("Compressed Training Accuracy:{0}".format(training_accuracy))
                        print("Compressed Test Accuracy:{0}".format(test_accuracy))
                        kv_table_key = "Leaf:{0} T:{1} slW:{2} hlW:{3} l2W:{4} lr:{5}".format(leaf_node.index,
                                                                                              temperature,
                                                                                              soft_loss_weight,
                                                                                              hard_loss_weight,
                                                                                              l2_weight, lr
                                                                                              )
                        kv_rows.append((run_id, iteration, "Training Accuracy {0}".format(kv_table_key),
                                        training_accuracy))
                        kv_rows.append((run_id, iteration, "Test Accuracy {0}".format(kv_table_key),
                                        test_accuracy))
                        break
            DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        print("X")

    @staticmethod
    def build_compressed_probabilities(network, leaf_node, posteriors, one_hot_labels):
        # Order mode labels from small to large, assign the smallest label to "0", next one to "1" and so on.
        # If there are N modes, Outlier class will have N as the label.
        label_count = one_hot_labels.shape[1]
        modes_per_leaves = network.modeTracker.get_modes()
        sorted_modes = sorted(modes_per_leaves[leaf_node.index])
        non_mode_labels = [l for l in range(label_count) if l not in modes_per_leaves[leaf_node.index]]
        mode_posteriors = posteriors[:, sorted_modes]
        outlier_posteriors = np.sum(posteriors[:, non_mode_labels], 1).reshape(posteriors.shape[0], 1)
        compressed_posteriors = np.concatenate((mode_posteriors, outlier_posteriors), axis=1)
        mode_one_hot_entries = one_hot_labels[:, sorted_modes]
        outlier_one_hot_entries = np.sum(one_hot_labels[:, non_mode_labels], 1).reshape(one_hot_labels.shape[0], 1)
        compressed_one_hot_entries = np.concatenate((mode_one_hot_entries, outlier_one_hot_entries), axis=1)
        print("X")
        return compressed_posteriors, compressed_one_hot_entries

    @staticmethod
    def calculate_compressed_accuracy(posteriors, one_hot_labels):
        assert posteriors.shape[0] == one_hot_labels.shape[0]
        posterior_max = np.argmax(posteriors, axis=1)
        one_hot_max = np.argmax(one_hot_labels, axis=1)
        correct_count = np.sum(posterior_max == one_hot_max)
        accuracy = float(correct_count) / float(posteriors.shape[0])
        # print("Accuracy:{0}".format(accuracy))
        return accuracy

    @staticmethod
    def get_tempered_probabilities(logits, temperature):
        tempered_logits = logits / temperature
        exp_logits = np.exp(tempered_logits)
        logit_sums = np.sum(exp_logits, 1).reshape(exp_logits.shape[0], 1)
        tempered_posteriors = exp_logits / logit_sums
        return tempered_posteriors
