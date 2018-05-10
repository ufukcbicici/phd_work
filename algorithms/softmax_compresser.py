import itertools
import threading
from random import shuffle

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants, SoftmaxCompressionStrategy, GradientType


class NetworkOutputs:
    def __init__(self):
        self.featureVectorsDict = {}
        self.logitsDict = {}
        self.posteriorsDict = {}
        self.oneHotLabelsDict = {}


class LogisticRegressionFitter(threading.Thread):
    def __init__(self, thread_id, reg_weights_list,
                 training_features, training_labels,
                 test_features, test_labels,
                 cross_val_count):
        threading.Thread.__init__(self)
        self.threadId = thread_id
        self.regWeights = reg_weights_list
        self.trainingFeatures = training_features
        self.trainingLabels = training_labels
        self.testFeatures = test_features
        self.testLabels = test_labels
        self.crossValidationCount = cross_val_count
        self.results = []

    def run(self):
        for regularizer_weight in self.regWeights:
            logistic_regression = LogisticRegression(solver="newton-cg", multi_class="multinomial",
                                                     C=regularizer_weight, max_iter=1000)
            cv_scores = cross_val_score(estimator=logistic_regression, X=self.trainingFeatures, y=self.trainingLabels,
                                        cv=self.crossValidationCount)
            logistic_regression.fit(X=self.trainingFeatures, y=self.trainingLabels)
            score_training = logistic_regression.score(X=self.trainingFeatures, y=self.trainingLabels)
            score_test = logistic_regression.score(X=self.testFeatures, y=self.testLabels)
            mean_score = np.mean(cv_scores)
            self.results.append((mean_score, logistic_regression, regularizer_weight, score_training, score_test))
            # print("L2 Weight:{0} Mean CV Score:{1} Training Score:{2} Test Score:{3}".format(regularizer_weight,
            #                                                                                  mean_score,
            #                                                                                  score_training,
            #                                                                                  score_test))


# class RunData:
#     def __init__(self):


class TfObjects:
    def __init__(self,
                 session,
                 global_step,
                 loss,
                 trainer,
                 learning_rate,
                 sm_weights,
                 sm_biases,
                 prob_tensor,
                 one_hot_tensor,
                 features_tensor,
                 compressed_softmax_output,
                 soft_cost_weight_tensor,
                 hard_cost_weight_tensor,
                 l2_loss_weight_tensor
                 ):
        self.session = session
        self.globalStep = global_step
        self.loss = loss
        self.trainer = trainer
        self.learningRate = learning_rate
        self.softmaxWeights = sm_weights
        self.softmaxBiases = sm_biases
        self.probTensor = prob_tensor
        self.oneHotTensor = one_hot_tensor
        self.featuresTensor = features_tensor
        self.compressedSoftmaxOutput = compressed_softmax_output
        self.softCostWeightTensor = soft_cost_weight_tensor
        self.hardCostWeightTensor = hard_cost_weight_tensor
        self.l2LossWeightTensor = l2_loss_weight_tensor


class DataObjects:
    def __init__(self, training_logits, test_logits,
                 training_one_hot_labels, test_one_hot_labels,
                 training_features, test_features):
        self.trainingLogits = training_logits
        self.testLogits = test_logits
        self.trainingOneHotLabels = training_one_hot_labels
        self.testOneHotLabels = test_one_hot_labels
        self.trainingFeatures = training_features
        self.testFeatures = test_features


class SoftmaxCompresser:
    def __init__(self, network, dataset, run_id):
        self.network = network
        self.dataset = dataset
        self.runId = run_id
        self.labelMappings = {}
        self.inverseLabelMappings = {}

    def compress_network_softmax(self, sess):
        # Get all final feature vectors for all leaves, for the complete training set.
        softmax_weights = {}
        softmax_biases = {}
        network_outputs = {}
        for dataset_type in [DatasetTypes.training, DatasetTypes.test, DatasetTypes.test]:
            network_output = NetworkOutputs()
            self.dataset.set_current_data_set_type(dataset_type=dataset_type)
            while True:
                results = self.network.eval_network(sess=sess, dataset=self.dataset, use_masking=True)
                for node in self.network.topologicalSortedNodes:
                    if not node.isLeaf:
                        continue
                    leaf_node = node
                    posterior_ref = self.network.get_variable_name(name="posterior_probs", node=leaf_node)
                    posterior_probs = results[posterior_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.posteriorsDict, key=leaf_node.index,
                                                         array=posterior_probs)
                    final_feature_ref = self.network.get_variable_name(name="final_eval_feature", node=leaf_node)
                    final_leaf_features = results[final_feature_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.featureVectorsDict, key=leaf_node.index,
                                                         array=final_leaf_features)
                    one_hot_label_ref = "Node{0}_one_hot_label_tensor".format(leaf_node.index)
                    one_hot_labels = results[one_hot_label_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.oneHotLabelsDict, key=leaf_node.index,
                                                         array=one_hot_labels)
                    logits_ref = self.network.get_variable_name(name="logits", node=leaf_node)
                    logits = results[logits_ref]
                    UtilityFuncs.concat_to_np_array_dict(dct=network_output.logitsDict, key=leaf_node.index,
                                                         array=logits)
                    softmax_weights_ref = self.network.get_variable_name(name="fc_softmax_weights", node=leaf_node)
                    softmax_weights[leaf_node.index] = results[softmax_weights_ref]
                    softmax_biases_ref = self.network.get_variable_name(name="fc_softmax_biases", node=leaf_node)
                    softmax_biases[leaf_node.index] = results[softmax_biases_ref]
                if self.dataset.isNewEpoch:
                    network_outputs[dataset_type] = network_output
                    break

        # Train all leaf classifiers by distillation
        compressed_layers_dict = {}
        self.labelMappings = {}
        self.network.variableManager.remove_variables_with_name(name="_softmax_")
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            leaf_node = node
            modes_per_leaves = self.network.modeTracker.get_modes()
            sorted_modes = sorted(modes_per_leaves[leaf_node.index])
            label_mapping, inverse_label_mapping = SoftmaxCompresser.get_compressed_probability_mapping(
                modes=sorted_modes,
                dataset=self.dataset)
            self.labelMappings[leaf_node.index] = label_mapping
            self.inverseLabelMappings[leaf_node.index] = inverse_label_mapping
            if GlobalConstants.SOFTMAX_COMPRESSION_STRATEGY == SoftmaxCompressionStrategy.fit_logistic_layer:
                logistic_weights, logistic_bias = self.train_logistic_layer(sess=sess,
                                                                            training_data=network_outputs[
                                                                                DatasetTypes.training],
                                                                            validation_data=
                                                                            network_outputs[
                                                                                DatasetTypes.test],
                                                                            test_data=network_outputs[
                                                                                DatasetTypes.test],
                                                                            leaf_node=leaf_node,
                                                                            cross_val_count=10)
                compressed_layers_dict[leaf_node.index] = (logistic_weights, logistic_bias)
            elif GlobalConstants.SOFTMAX_COMPRESSION_STRATEGY == SoftmaxCompressionStrategy.random_start:
                logistic_weights, logistic_bias = self.init_random_logistic_layer(sess=sess,
                                                                                  training_data=
                                                                                  network_outputs[
                                                                                      DatasetTypes.training],
                                                                                  leaf_node=leaf_node)
                compressed_layers_dict[leaf_node.index] = (logistic_weights, logistic_bias)
            else:
                raise NotImplementedError()
        # Change the loss layers for all leaf nodes
        for node in self.network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            leaf_node = node
            self.change_leaf_loss(node=leaf_node, compressed_layers_dict=compressed_layers_dict)
        # Redefine the main classification loss and the regularization loss; these are dependent on the new
        # compressed hyperplanes.
        self.network.build_main_loss()
        self.network.build_regularization_loss()
        # Re-calculate the gradients
        GlobalConstants.GRADIENT_TYPE = GlobalConstants.SOFTMAX_DISTILLATION_GRADIENT_TYPE
        self.network.gradFunc(network=self.network)

    def change_leaf_loss(self, node, compressed_layers_dict):
        softmax_weights = compressed_layers_dict[node.index][0]
        softmax_biases = compressed_layers_dict[node.index][1]
        logits = tf.matmul(node.finalFeatures, softmax_weights) + softmax_biases
        self.network.evalDict[self.network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        if node.labelMappingTensor is None:
            node.labelMappingTensor = tf.placeholder(name="label_mapping_node_{0}".format(node.index), dtype=tf.int64)
            node.compressedLabelsTensor = tf.nn.embedding_lookup(params=node.labelMappingTensor, ids=node.labelTensor)
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.compressedLabelsTensor,
                                                                                   logits=logits)
        parallel_dnn_updates = {GradientType.parallel_dnns_unbiased, GradientType.parallel_dnns_biased}
        mixture_of_expert_updates = {GradientType.mixture_of_experts_biased, GradientType.mixture_of_experts_unbiased}
        if GlobalConstants.SOFTMAX_DISTILLATION_GRADIENT_TYPE in parallel_dnn_updates:
            pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
            loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        elif GlobalConstants.SOFTMAX_DISTILLATION_GRADIENT_TYPE in mixture_of_expert_updates:
            pre_loss = tf.reduce_sum(cross_entropy_loss_tensor)
            loss = (1.0 / float(GlobalConstants.BATCH_SIZE)) * pre_loss
        else:
            raise NotImplementedError()
        node.fOpsList[-1] = loss
        assert len(node.lossList) == 1
        node.lossList = [loss]

    # def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
    #     final_feature_final = final_feature
    #     if GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION:
    #         final_feature_final = tf.nn.dropout(final_feature, self.classificationDropoutKeepProb)
    #     if GlobalConstants.USE_DECISION_AUGMENTATION:
    #         concat_list = [final_feature_final]
    #         concat_list.extend(node.activationsDict.values())
    #         final_feature_final = tf.concat(values=concat_list, axis=1)
    #     node.residueOutputTensor = final_feature_final
    #     node.finalFeatures = final_feature_final
    #     node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature_final
    #     node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature_final)
    #     logits = tf.matmul(final_feature_final, softmax_weights) + softmax_biases
    #     cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
    #                                                                                logits=logits)
    #     parallel_dnn_updates = {GradientType.parallel_dnns_unbiased, GradientType.parallel_dnns_biased}
    #     mixture_of_expert_updates = {GradientType.mixture_of_experts_biased, GradientType.mixture_of_experts_unbiased}
    #     if GlobalConstants.GRADIENT_TYPE in parallel_dnn_updates:
    #         pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
    #         loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
    #     elif GlobalConstants.GRADIENT_TYPE in mixture_of_expert_updates:
    #         pre_loss = tf.reduce_sum(cross_entropy_loss_tensor)
    #         loss = (1.0 / float(GlobalConstants.BATCH_SIZE)) * pre_loss
    #     else:
    #         raise NotImplementedError()
    #     node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
    #     node.lossList.append(loss)
    #     return final_feature_final, logits

    # def apply_compressed_loss(self, node, softmax_weights, softmax_biases):





    @staticmethod
    def assert_prob_correctness(softmax_weights, softmax_biases, features, logits, probs, leaf_node):
        print("max features entry:{0}".format(np.max(features)))
        print("max softmax_weights entry:{0}".format(np.max(softmax_weights)))
        print("min softmax_weights entry:{0}".format(np.min(softmax_weights)))
        print("max softmax_biases entry:{0}".format(np.max(softmax_biases)))
        print("min softmax_biases entry:{0}".format(np.min(softmax_biases)))
        npz_file_name = "npz_node_{0}_distillation".format(leaf_node.index)
        # UtilityFuncs.save_npz(npz_file_name,
        #                       arr_dict={"softmax_weights": softmax_weights,
        #                                 "softmax_biases": softmax_biases,
        #                                 "features": features,
        #                                 "logits": logits, "probs": probs})
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

    def train_distillation_network(self, sess, leaf_node, training_data, test_data):
        training_logits = training_data.logitsDict[leaf_node.index]
        training_one_hot_labels = training_data.oneHotLabelsDict[leaf_node.index]
        training_features = training_data.featureVectorsDict[leaf_node.index]
        test_logits = test_data.logitsDict[leaf_node.index]
        test_one_hot_labels = test_data.oneHotLabelsDict[leaf_node.index]
        test_features = test_data.featureVectorsDict[leaf_node.index]
        assert training_logits.shape[0] == training_one_hot_labels.shape[0]
        assert training_logits.shape[0] == training_features.shape[0]
        # Build the tempered posteriors
        training_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=training_logits,
                                                                                    temperature=1.0)
        test_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=test_logits,
                                                                                temperature=1.0)
        # Get the compressed probabilities
        training_compressed_posteriors, training_compressed_one_hot_entries = \
            self.build_compressed_probabilities(leaf_node=leaf_node,
                                                posteriors=training_tempered_posteriors,
                                                one_hot_labels=training_one_hot_labels)
        test_compressed_posteriors, test_compressed_one_hot_entries = \
            self.build_compressed_probabilities(leaf_node=leaf_node,
                                                posteriors=test_tempered_posteriors,
                                                one_hot_labels=test_one_hot_labels)
        logit_dim = training_logits.shape[1]
        features_dim = training_features.shape[1]
        modes_per_leaves = self.network.modeTracker.get_modes()
        compressed_class_count = len(modes_per_leaves[leaf_node.index]) + 1
        npz_file_name = "npz_node_{0}_final_features".format(leaf_node.index)
        UtilityFuncs.save_npz(npz_file_name,
                              arr_dict={"training_features": training_features,
                                        "training_one_hot_labels": training_compressed_one_hot_entries,
                                        "training_compressed_posteriors": training_compressed_posteriors,
                                        "training_logits": training_logits,
                                        "leaf_modes": np.array(modes_per_leaves[leaf_node.index]),
                                        "test_features": test_features,
                                        "test_one_hot_labels": test_compressed_one_hot_entries,
                                        "test_compressed_posteriors": test_compressed_posteriors,
                                        "test_logits": test_logits})
        # p: The tempered posteriors, which have been squashed.
        p = tf.placeholder(tf.float32, shape=(None, compressed_class_count))
        # t: The squashed one hot labels
        t = tf.placeholder(tf.float32, shape=(None, compressed_class_count))
        features_tensor = tf.placeholder(tf.float32, shape=(None, features_dim))
        soft_labels_cost_weight = tf.placeholder(tf.float32)
        hard_labels_cost_weight = tf.placeholder(tf.float32)
        l2_loss_weight = tf.placeholder(tf.float32)
        # Get new class count: Mode labels + Outliers. Init the new classifier hyperplanes.
        softmax_weights = self.network.variableManager.create_and_add_variable_to_node(
            node=leaf_node,
            name=self.network.get_variable_name(name="distilled_fc_softmax_weights_{0}".format(self.runId)),
            initial_value=tf.truncated_normal([features_dim, compressed_class_count],
                                              stddev=0.1,
                                              seed=GlobalConstants.SEED,
                                              dtype=GlobalConstants.DATA_TYPE),
            is_trainable=True
        )
        softmax_biases = self.network.variableManager.create_and_add_variable_to_node(
            node=leaf_node,
            name=self.network.get_variable_name(name="distilled_fc_softmax_biases_{0}".format(self.runId)),
            initial_value=tf.constant(0.1, shape=[compressed_class_count], dtype=GlobalConstants.DATA_TYPE),
            is_trainable=True
        )
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
        # grad_soft_loss = tf.gradients(ys=soft_loss, xs=[softmax_weights, softmax_biases])
        # grad_hard_loss = tf.gradients(ys=hard_loss, xs=[softmax_weights, softmax_biases])
        # grad_sm_weights = tf.gradients(ys=weight_l2, xs=[softmax_weights])
        # Counter
        global_step = self.network.variableManager.create_and_add_variable_to_node(node=None, name="global_step",
                                                                                   initial_value=0,
                                                                                   is_trainable=False)
        # Train by cross-validation
        # temperature_list = [1.0]
        # soft_loss_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        # hard_loss_weights = [1.0]
        # l2_weights = [0.0, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
        # learning_rates = [0.001, 0.005, 0.025, 0.05, 0.075, 0.1]
        temperature_list = [1.0]
        soft_loss_weights = [0.0]
        hard_loss_weights = [1.0]
        l2_weights = [0.0]
        learning_rates = [0.005]
        cross_validation_repeat_count = 10
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[learning_rates,
                                                                              temperature_list, soft_loss_weights,
                                                                              hard_loss_weights,
                                                                              l2_weights])
        duplicate_cartesians = []
        for tpl in cartesian_product:
            duplicate_cartesians.extend(list(itertools.repeat(tpl, cross_validation_repeat_count)))
        results_dict = {}
        params_dict = {}
        # Tensorflow flow, computation graph objects
        tf_object = TfObjects(session=sess,
                              global_step=global_step,
                              loss=distillation_loss,
                              trainer=None,
                              learning_rate=None,
                              sm_weights=softmax_weights,
                              sm_biases=softmax_biases,
                              prob_tensor=p,
                              one_hot_tensor=t,
                              features_tensor=features_tensor,
                              compressed_softmax_output=compressed_softmax_output,
                              soft_cost_weight_tensor=soft_labels_cost_weight,
                              hard_cost_weight_tensor=hard_labels_cost_weight,
                              l2_loss_weight_tensor=l2_loss_weight)
        # Training and Test data
        data_object = DataObjects(training_logits=training_logits,
                                  test_logits=test_logits,
                                  training_one_hot_labels=training_one_hot_labels,
                                  test_one_hot_labels=test_one_hot_labels,
                                  training_features=training_features,
                                  test_features=test_features)
        # Cross Validation
        curr_lr = 0.0
        # A new run for each tuple
        for tpl in duplicate_cartesians:
            lr = tpl[0]
            temperature = tpl[1]
            soft_loss_weight = tpl[2]
            hard_loss_weight = tpl[3]
            l2_weight = tpl[4]
            kv_rows = []
            # Build the optimizer
            if curr_lr != lr:
                tf_object.learningRate = tf.train.exponential_decay(lr, global_step,
                                                                    GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT,
                                                                    GlobalConstants.SOFTMAX_DISTILLATION_DECAY,
                                                                    staircase=True)
                tf_object.trainer = tf.train.MomentumOptimizer(tf_object.learningRate, 0.9) \
                    .minimize(distillation_loss,
                              global_step=global_step)
                curr_lr = lr
            # Run the algorithm
            final_softmax_weights, final_softmax_biases, final_training_accuracy, final_test_accuracy = \
                self.train_for_params(leaf_node=leaf_node,
                                      lr=lr, temperature=temperature,
                                      soft_loss_weight=soft_loss_weight,
                                      hard_loss_weight=hard_loss_weight, l2_weight=l2_weight,
                                      tf_objects=tf_object, data_objects=data_object, kv_rows=kv_rows)
            if GlobalConstants.USE_SOFTMAX_DISTILLATION_VERBOSE:
                DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        # # Pick the best result
        # averaged_results = []
        # for k, v in results_dict.items():
        #     averaged_results.append((k, sum(v) / len(v)))
        # sorted_results = sorted(averaged_results, key=lambda pair: pair[1], reverse=True)
        # best_result = sorted_results[0]
        # best_params = best_result[0]
        # # Get the best result among the best parameters
        print("X")

    def train_for_params(self, leaf_node,
                         lr, temperature, soft_loss_weight, hard_loss_weight, l2_weight,
                         tf_objects, data_objects, kv_rows):
        assert tf_objects.trainer is not None and tf_objects.learningRate is not None
        # Init variables
        all_variables = tf.global_variables()
        vars_to_init = [var for var in all_variables if "/Momentum" in var.name]
        vars_to_init.extend([tf_objects.softmaxWeights, tf_objects.softmaxBiases, tf_objects.globalStep])
        # Init variables
        init_op = tf.variables_initializer(vars_to_init)
        # Build the tempered posteriors
        training_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=data_objects.trainingLogits,
                                                                                    temperature=temperature)
        test_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=data_objects.testLogits,
                                                                                temperature=temperature)
        # Get the compressed probabilities
        training_compressed_posteriors, training_compressed_one_hot_entries = \
            self.build_compressed_probabilities(leaf_node=leaf_node,
                                                posteriors=training_tempered_posteriors,
                                                one_hot_labels=data_objects.trainingOneHotLabels)
        test_compressed_posteriors, test_compressed_one_hot_entries = \
            self.build_compressed_probabilities(leaf_node=leaf_node,
                                                posteriors=test_tempered_posteriors,
                                                one_hot_labels=data_objects.testOneHotLabels)

        # Training sets
        training_sample_count = data_objects.trainingFeatures.shape[0]
        training_indices = list(range(training_sample_count))
        shuffle(training_indices)
        random_indices = np.random.uniform(0, training_sample_count,
                                           GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE).astype(int).tolist()
        training_p = training_compressed_posteriors[training_indices]
        training_t = training_compressed_one_hot_entries[training_indices]
        training_x = data_objects.trainingFeatures[training_indices]
        max_feature_entry = np.max(training_x)
        min_feature_entry = np.min(training_x)
        kv_rows.append((self.runId, -1,
                        "Leaf:{0} max_feature_entry".format(leaf_node.index), max_feature_entry))
        kv_rows.append((self.runId, -1,
                        "Leaf:{0} min_feature_entry".format(leaf_node.index), min_feature_entry))
        training_indices.extend(random_indices)
        training_p_wrapped = training_compressed_posteriors[training_indices]
        training_t_wrapped = training_compressed_one_hot_entries[training_indices]
        training_x_wrapped = data_objects.trainingFeatures[training_indices]
        # Test sets
        test_sample_count = data_objects.testFeatures.shape[0]
        test_indices = list(range(test_sample_count))
        test_p = test_compressed_posteriors[test_indices]
        test_t = test_compressed_one_hot_entries[test_indices]
        test_x = data_objects.testFeatures[test_indices]
        # Calculate accuracy on the training set
        training_accuracy_full = \
            SoftmaxCompresser.calculate_compressed_accuracy(posteriors=training_p, one_hot_labels=training_t)
        # Calculate accuracy on the validation set
        test_accuracy_full = \
            SoftmaxCompresser.calculate_compressed_accuracy(posteriors=test_p, one_hot_labels=test_t)
        kv_rows.append((self.runId, -1,
                        "Leaf:{0} Training Accuracy Full".format(leaf_node.index), training_accuracy_full))
        kv_rows.append((self.runId, -1,
                        "Leaf:{0} Test Accuracy Full".format(leaf_node.index), test_accuracy_full))
        # Train
        batch_size = int(float(training_sample_count) * GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO)
        # GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE
        # Init softmax parameters
        tf_objects.session.run(init_op)
        iteration = 0
        for epoch_id in range(GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT):
            curr_index = 0
            while True:
                p_batch = training_p_wrapped[curr_index:curr_index + batch_size]
                t_batch = training_t_wrapped[curr_index:curr_index + batch_size]
                features_batch = training_x_wrapped[curr_index:curr_index + batch_size]
                feed_dict = {tf_objects.probTensor: p_batch,
                             tf_objects.oneHotTensor: t_batch,
                             tf_objects.featuresTensor: features_batch,
                             tf_objects.softCostWeightTensor: soft_loss_weight,
                             tf_objects.hardCostWeightTensor: hard_loss_weight,
                             tf_objects.l2LossWeightTensor: l2_weight}
                # run_ops = [grad_soft_loss,
                #            grad_hard_loss,
                #            grad_sm_weights,
                #            trainer,
                #            learning_rate]
                run_ops = [tf_objects.trainer, tf_objects.learningRate]
                results = tf_objects.session.run(run_ops, feed_dict=feed_dict)
                iteration += 1
                print("Iteration:{0} Learning Rate:{1}".format(iteration, results[-1]))
                # grad_soft_loss_weight_mag = np.linalg.norm(results[0][0])
                # grad_soft_loss_bias_mag = np.linalg.norm(results[0][1])
                # grad_hard_loss_weight_mag = np.linalg.norm(results[1][0])
                # grad_hard_loss_bias_mag = np.linalg.norm(results[1][1])
                # print("grad_soft_loss_weight_mag={0}".format(grad_soft_loss_weight_mag))
                # print("grad_soft_loss_bias_mag={0}".format(grad_soft_loss_bias_mag))
                # print("grad_hard_loss_weight_mag={0}".format(grad_hard_loss_weight_mag))
                # print("grad_hard_loss_bias_mag={0}".format(grad_hard_loss_bias_mag))
                curr_index += batch_size
                if curr_index >= training_sample_count:
                    is_last_epoch = epoch_id == GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT - 1
                    # Evaluate on training set
                    training_results = tf_objects.session.run(
                        [tf_objects.compressedSoftmaxOutput, tf_objects.loss,
                         tf_objects.softmaxWeights, tf_objects.softmaxBiases],
                        feed_dict={tf_objects.probTensor: training_p,
                                   tf_objects.oneHotTensor: training_t,
                                   tf_objects.featuresTensor: training_x,
                                   tf_objects.softCostWeightTensor: soft_loss_weight,
                                   tf_objects.hardCostWeightTensor: hard_loss_weight,
                                   tf_objects.l2LossWeightTensor: l2_weight})
                    training_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                        posteriors=training_results[0], one_hot_labels=training_t)
                    # Evaluate on test set
                    test_results = tf_objects.session.run(
                        [tf_objects.compressedSoftmaxOutput, tf_objects.loss],
                        feed_dict={tf_objects.probTensor: test_p,
                                   tf_objects.oneHotTensor: test_t,
                                   tf_objects.featuresTensor: test_x,
                                   tf_objects.softCostWeightTensor: soft_loss_weight,
                                   tf_objects.hardCostWeightTensor: hard_loss_weight,
                                   tf_objects.l2LossWeightTensor: l2_weight})
                    test_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                        posteriors=test_results[0], one_hot_labels=test_t)
                    # Get resulting linear classifiers
                    hyperplane_weights = training_results[2]
                    hyperplane_biases = training_results[3]
                    print("Uncompressed Training Accuracy:{0}".format(training_accuracy_full))
                    print("Uncompressed Test Accuracy:{0}".format(test_accuracy_full))
                    print("Compressed Training Accuracy:{0}".format(training_accuracy))
                    print("Compressed Test Accuracy:{0}".format(test_accuracy))
                    if GlobalConstants.USE_SOFTMAX_DISTILLATION_VERBOSE:
                        kv_table_key = "Leaf:{0} T:{1} slW:{2} hlW:{3} l2W:{4} lr:{5}".format(leaf_node.index,
                                                                                              temperature,
                                                                                              soft_loss_weight,
                                                                                              hard_loss_weight,
                                                                                              l2_weight, lr
                                                                                              )
                        kv_rows.append((self.runId, iteration, "Training Accuracy {0}".format(kv_table_key),
                                        training_accuracy))
                        kv_rows.append((self.runId, iteration, "Test Accuracy {0}".format(kv_table_key),
                                        test_accuracy))
                    if is_last_epoch:
                        final_softmax_weights = hyperplane_weights
                        final_softmax_biases = hyperplane_biases
                        final_training_accuracy = training_accuracy
                        final_test_accuracy = test_accuracy
                        return final_softmax_weights, final_softmax_biases, final_training_accuracy, final_test_accuracy
                    break

    def build_compressed_probabilities(self, leaf_node, posteriors, one_hot_labels):
        # Order mode labels from small to large, assign the smallest label to "0", next one to "1" and so on.
        # If there are N modes, Outlier class will have N as the label.
        label_count = one_hot_labels.shape[1]
        modes_per_leaves = self.network.modeTracker.get_modes()
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
    def get_compressed_probability_mapping(modes, dataset):
        label_count = dataset.get_label_count()
        sorted_modes = sorted(modes)
        label_mapping = np.zeros(shape=(label_count,), dtype=np.int32)
        inverse_label_mapping = {}
        for i in range(len(sorted_modes)):
            label_mapping[sorted_modes[i]] = i
            inverse_label_mapping[i] = sorted_modes[i]
        inverse_label_mapping[len(sorted_modes)] = -1
        for l in range(label_count):
            if l not in sorted_modes:
                label_mapping[l] = len(sorted_modes)
        return label_mapping, inverse_label_mapping

    @staticmethod
    def compress_probability(modes, probability):
        dim = probability.shape[1]
        sorted_modes = sorted(modes)
        non_mode_labels = [l for l in range(dim) if l not in modes]
        mode_probs = probability[:, sorted_modes]
        outlier_probs = np.sum(probability[:, non_mode_labels], 1).reshape(probability.shape[0], 1)
        compressed_probs = np.concatenate((mode_probs, outlier_probs), axis=1)
        return compressed_probs

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
        max_logits = np.max(tempered_logits, axis=1).reshape(tempered_logits.shape[0], 1)
        exp_logits = np.exp(tempered_logits - max_logits)
        logit_sums = np.sum(exp_logits, 1).reshape(exp_logits.shape[0], 1)
        tempered_posteriors = exp_logits / logit_sums
        return tempered_posteriors

    def train_logistic_layer(self, sess, training_data, validation_data, test_data, leaf_node,
                             cross_val_count):
        modes_per_leaves = self.network.modeTracker.get_modes()
        sorted_modes = sorted(modes_per_leaves[leaf_node.index])
        training_logits = training_data.logitsDict[leaf_node.index]
        training_one_hot_labels = training_data.oneHotLabelsDict[leaf_node.index]
        training_features = training_data.featureVectorsDict[leaf_node.index]
        training_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=training_logits, temperature=1.0)
        training_posteriors_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                probability=training_posteriors)
        training_one_hot_labels_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                    probability=training_one_hot_labels)
        training_labels = np.argmax(training_one_hot_labels_compressed, axis=1)

        validation_logits = validation_data.logitsDict[leaf_node.index]
        validation_one_hot_labels = validation_data.oneHotLabelsDict[leaf_node.index]
        validation_features = validation_data.featureVectorsDict[leaf_node.index]
        validation_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=validation_logits, temperature=1.0)
        validation_posteriors_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                  probability=validation_posteriors)
        validation_one_hot_labels_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                      probability=validation_one_hot_labels)
        validation_labels = np.argmax(validation_one_hot_labels_compressed, axis=1)

        test_logits = test_data.logitsDict[leaf_node.index]
        test_one_hot_labels = test_data.oneHotLabelsDict[leaf_node.index]
        test_features = test_data.featureVectorsDict[leaf_node.index]
        test_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=test_logits, temperature=1.0)
        test_posteriors_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                            probability=test_posteriors)
        test_one_hot_labels_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                probability=test_one_hot_labels)
        test_labels = np.argmax(test_one_hot_labels_compressed, axis=1)

        # Calculate accuracy on the training set
        training_accuracy_full = \
            SoftmaxCompresser.calculate_compressed_accuracy(posteriors=training_posteriors_compressed,
                                                            one_hot_labels=training_one_hot_labels_compressed)
        # Calculate accuracy on the validation set
        validation_accuracy_full = \
            SoftmaxCompresser.calculate_compressed_accuracy(posteriors=validation_posteriors_compressed,
                                                            one_hot_labels=validation_one_hot_labels_compressed)
        # Calculate accuracy on the test set
        test_accuracy_full = \
            SoftmaxCompresser.calculate_compressed_accuracy(posteriors=test_posteriors_compressed,
                                                            one_hot_labels=test_one_hot_labels_compressed)
        print("training_accuracy_full={0}".format(training_accuracy_full))
        print("validation_accuracy_full={0}".format(validation_accuracy_full))
        print("test_accuracy_full={0}".format(test_accuracy_full))

        regularizer_weights = [0.00001, 0.000025, 0.00005,
                               0.0001, 0.00025, 0.0005,
                               0.001, 0.0025, 0.005,
                               0.01, 0.025, 0.05,
                               0.1, 0.25, 0.5,
                               1.0, 1.25, 1.5,
                               2.5, 3.0, 3.5,
                               4.5, 5.0, 10.0, 15.0, 20.0, 25.0,
                               100.0, 250.0, 500.0,
                               1000.0, 2500.0, 5000.0, 10000.0]
        # regularizer_weights = [0.01]

        results_list = []
        best_test_result = 0.0
        best_test_l2_weight = -1.0
        regularizer_weights_dict = \
            UtilityFuncs.distribute_evenly_to_threads(num_of_threads=GlobalConstants.SOFTMAX_DISTILLATION_CPU_COUNT,
                                                      list_to_distribute=regularizer_weights)
        threads_dict = {}
        for thread_id in range(GlobalConstants.SOFTMAX_DISTILLATION_CPU_COUNT):
            threads_dict[thread_id] = LogisticRegressionFitter(thread_id=thread_id,
                                                               reg_weights_list=regularizer_weights_dict[thread_id],
                                                               training_features=training_features,
                                                               training_labels=training_labels,
                                                               test_features=test_features, test_labels=test_labels,
                                                               cross_val_count=cross_val_count)
            threads_dict[thread_id].start()
        all_results = []
        for thread in threads_dict.values():
            thread.join()
        for thread in threads_dict.values():
            all_results.extend(thread.results)
        sorted_results_best_validation = sorted(all_results, key=lambda tpl: tpl[0], reverse=True)
        sorted_results_best_test = sorted(all_results, key=lambda tpl: tpl[-1], reverse=True)
        best_result_validation = sorted_results_best_validation[0]
        best_test_result = sorted_results_best_test[0][-1]
        selected_logistic_model = best_result_validation[1]
        features_dim = training_features.shape[1]
        compressed_class_count = training_one_hot_labels_compressed.shape[1]
        logistic_weight = np.transpose(selected_logistic_model.coef_)
        logistic_bias = selected_logistic_model.intercept_
        softmax_weights = self.network.variableManager.create_and_add_variable_to_node(
            node=leaf_node,
            initial_value=tf.constant(logistic_weight, dtype=GlobalConstants.DATA_TYPE),
            name=self.network.get_variable_name(name="distilled_fc_softmax_weights_{0}".format(self.runId),
                                                node=leaf_node))
        softmax_biases = self.network.variableManager.create_and_add_variable_to_node(
            node=leaf_node,
            initial_value=tf.constant(logistic_bias, dtype=GlobalConstants.DATA_TYPE),
            name=self.network.get_variable_name(name="distilled_fc_softmax_biases_{0}".format(self.runId),
                                                node=leaf_node))
        sess.run([softmax_weights.initializer, softmax_biases.initializer])
        print("Best Test Result:{0} Best L2 Weight:{1}".format(best_test_result, best_test_l2_weight))
        print("Selected L2 Weight:{0}".format(best_result_validation[2]))
        if GlobalConstants.SOFTMAX_DISTILLATION_VERBOSE:
            x_tensor = tf.placeholder(tf.float32)
            compressed_logits = tf.matmul(x_tensor, softmax_weights) + softmax_biases
            result = sess.run([compressed_logits], feed_dict={x_tensor: training_features})
            tensorflow_response = result[0]
            scilearn_response = selected_logistic_model.decision_function(training_features)
            # assert np.allclose(tensorflow_response, scilearn_response)
            diff_matrix = np.abs(tensorflow_response - scilearn_response)
            ind = np.unravel_index(np.argmax(diff_matrix, axis=None), diff_matrix.shape)
            print("Most different tensorflow entry:{0}".format(tensorflow_response[ind]))
            print("Most different scilearn entry:{0}".format(scilearn_response[ind]))
        logistic_weight = np.transpose(selected_logistic_model.coef_)
        logistic_bias = selected_logistic_model.intercept_
        return softmax_weights, softmax_biases

    def init_random_logistic_layer(self, sess, training_data, leaf_node):
        modes_per_leaves = self.network.modeTracker.get_modes()
        sorted_modes = sorted(modes_per_leaves[leaf_node.index])
        training_logits = training_data.logitsDict[leaf_node.index]
        training_one_hot_labels = training_data.oneHotLabelsDict[leaf_node.index]
        training_features = training_data.featureVectorsDict[leaf_node.index]
        training_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=training_logits, temperature=1.0)
        training_posteriors_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                probability=training_posteriors)
        training_one_hot_labels_compressed = SoftmaxCompresser.compress_probability(modes=sorted_modes,
                                                                                    probability=training_one_hot_labels)
        training_labels = np.argmax(training_one_hot_labels_compressed, axis=1)

        features_dim = training_features.shape[1]
        compressed_class_count = training_one_hot_labels_compressed.shape[1]
        softmax_weights = self.network.variableManager.create_and_add_variable_to_node(
            node=leaf_node,
            initial_value=tf.truncated_normal([features_dim, compressed_class_count],
                                              stddev=0.1,
                                              seed=GlobalConstants.SEED,
                                              dtype=GlobalConstants.DATA_TYPE),
            name=self.network.get_variable_name(name="distilled_fc_softmax_weights_{0}".format(self.runId),
                                                node=leaf_node))
        softmax_biases = self.network.variableManager.create_and_add_variable_to_node(
            node=leaf_node,
            initial_value=tf.constant(0.1, shape=[compressed_class_count], dtype=GlobalConstants.DATA_TYPE),
            name=self.network.get_variable_name(name="distilled_fc_softmax_biases_{0}".format(self.runId),
                                                node=leaf_node))
        sess.run([softmax_weights.initializer, softmax_biases.initializer])
        return softmax_weights, softmax_biases
