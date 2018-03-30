from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import tensorflow as tf

from simple_tf.global_params import GlobalConstants


class SoftmaxCompresser:
    def __init__(self):
        pass

    @staticmethod
    def compress_network_softmax(network, sess, dataset):
        # Get all final feature vectors for all leaves, for the complete training set.
        feature_vectors_dict = {}
        logits_dict = {}
        posteriors_dict = {}
        one_hot_labels_dict = {}
        softmax_weights = {}
        softmax_biases = {}
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
        while True:
            results = network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            for node in network.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                leaf_node = node
                posterior_ref = network.get_variable_name(name="posterior_probs", node=leaf_node)
                posterior_probs = results[posterior_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=posteriors_dict, key=leaf_node.index, array=posterior_probs)
                final_feature_ref = network.get_variable_name(name="final_eval_feature", node=leaf_node)
                final_leaf_features = results[final_feature_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=feature_vectors_dict, key=leaf_node.index,
                                                     array=final_leaf_features)
                one_hot_label_ref = "Node{0}_one_hot_label_tensor".format(leaf_node.index)
                one_hot_labels = results[one_hot_label_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=one_hot_labels_dict, key=leaf_node.index,
                                                     array=one_hot_labels)
                logits_ref = network.get_variable_name(name="logits", node=leaf_node)
                logits = results[logits_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=logits_dict, key=leaf_node.index,
                                                     array=logits)
                softmax_weights_ref = network.get_variable_name(name="fc_softmax_weights", node=leaf_node)
                softmax_weights[leaf_node.index] = results[softmax_weights_ref]
                softmax_biases_ref = network.get_variable_name(name="fc_softmax_biases", node=leaf_node)
                softmax_biases[leaf_node.index] = results[softmax_biases_ref]

            if dataset.isNewEpoch:
                break
        # Train all leaf classifiers by distillation
        for node in network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            leaf_node = node
            softmax_weight = softmax_weights[leaf_node.index]
            softmax_bias = softmax_biases[leaf_node.index]
            feature_vectors = feature_vectors_dict[leaf_node.index]
            logits = logits_dict[leaf_node.index]
            probs = posteriors_dict[leaf_node.index]
            one_hot_labels = one_hot_labels_dict[leaf_node.index]
            # Unit Test
            SoftmaxCompresser.assert_prob_correctness(softmax_weights=softmax_weight, softmax_biases=softmax_bias,
                                                      features=feature_vectors, logits=logits, probs=probs)
            # Train compresser
            SoftmaxCompresser.train_distillation_network(network=network, leaf_node=leaf_node, logits=logits,
                                                         one_hot_labels=one_hot_labels, features=feature_vectors)


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
    def train_distillation_network(network, leaf_node, logits, one_hot_labels, features):
        assert logits.shape[0] == one_hot_labels.shape[0]
        assert logits.shape[0] == features.shape[0]
        logit_dim = logits.shape[1]
        features_dim = features.shape[1]
        # p: The tempered posteriors, which have been squashed.
        p = tf.placeholder(tf.float32, shape=(None, logit_dim))
        # t: The squashed one hot labels
        t = tf.placeholder(tf.float32, shape=(None, logit_dim))
        features_tensor = tf.placeholder(tf.float32, shape=(None, features_dim))
        soft_labels_cost_weight = tf.placeholder(tf.float32)
        hard_labels_cost_weight = tf.placeholder(tf.float32)
        l2_loss_weight = tf.placeholder(tf.float32)
        modes_per_leaves = network.modeTracker.get_modes()
        compressed_class_count = len(modes_per_leaves[leaf_node.index]) + 1
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
        # Gradients (For debug purposes)
        grad_soft_loss  = tf.gradients(ys=soft_loss, xs=[softmax_weights, softmax_biases])
        grad_hard_loss  = tf.gradients(ys=hard_loss, xs=[softmax_weights, softmax_biases])
        grad_sm_weights = tf.gradients(ys=weight_l2, xs=[softmax_weights])
        # Optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(GlobalConstants.SOFTMAX_DISTILLATION_INITIAL_LR, global_step,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_DECAY, staircase=True)
        trainer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(distillation_loss, global_step=global_step)
        # Train by cross-validation
        temperature_list = [1.0]
        soft_loss_weights = [1.0]
        hard_loss_weights = [1.0]
        l2_weights = [0.0]
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[temperature_list, soft_loss_weights,
                                                                              hard_loss_weights,
                                                                              l2_weights])
        for tpl in cartesian_product:
            temperature = tpl[0]
            soft_loss_weight = tpl[1]
            hard_loss_weight = tpl[2]
            l2_weight = tpl[3]
            # Build the tempered posteriors
            tempered_logits = logits / temperature
            exp_logits = np.exp(tempered_logits)
            logit_sums = np.sum(exp_logits, 1).reshape(exp_logits.shape[0], 1)
            tempered_posteriors = exp_logits / logit_sums
            # Get the compressed probabilities
            compressed_posteriors, compressed_one_hot_entries = \
                SoftmaxCompresser.build_compressed_probabilities(network=network, leaf_node=leaf_node,
                                                                 posteriors=tempered_posteriors,
                                                                 one_hot_labels=one_hot_labels)

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



