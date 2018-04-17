from algorithms.softmax_compresser import SoftmaxCompresser
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import tensorflow as tf
import itertools
from random import shuffle, random
from sklearn.preprocessing import RobustScaler

from simple_tf.global_params import GlobalConstants

# class_count = 3
# features_dim = 64
# node_index = 5


# node_5_features_dict = UtilityFuncs.load_npz(file_name="npz_node_5_final_features")
#
# training_features = node_5_features_dict["training_features"]
# training_one_hot_labels = node_5_features_dict["training_one_hot_labels"]
# training_compressed_posteriors = node_5_features_dict["training_compressed_posteriors"]
#
# test_features = node_5_features_dict["test_features"]
# test_one_hot_labels = node_5_features_dict["test_one_hot_labels"]
# test_compressed_posteriors = node_5_features_dict["test_compressed_posteriors"]

# class_count = 4
# features_dim = 64
node_index = 3
run_id = 4
modes = set()
modes.add(8)
modes.add(9)
hidden_feature_dim = 128

node_3_features_dict = UtilityFuncs.load_npz(file_name="npz_node_3_final_features")

training_features = node_3_features_dict["training_features"]
training_one_hot_labels = node_3_features_dict["training_one_hot_labels"]
training_compressed_posteriors = node_3_features_dict["training_compressed_posteriors"]
training_logits = node_3_features_dict["training_logits"]

test_features = node_3_features_dict["test_features"]
test_one_hot_labels = node_3_features_dict["test_one_hot_labels"]
test_compressed_posteriors = node_3_features_dict["test_compressed_posteriors"]
test_logits = node_3_features_dict["test_logits"]

features_dim = training_features.shape[1]
class_count = training_one_hot_labels.shape[1]
# data_scaler = RobustScaler()
# normalized_training_features = data_scaler.fit_transform(training_features)
# normalized_test_features = data_scaler.transform(test_features)

# training_features = normalized_training_features
# test_features = normalized_test_features

# for temperature in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 25.0, 50.0]:
#     print("Temperature:{0}".format(temperature))
#     training_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=training_logits,
#                                                                                 temperature=temperature)
#     print(training_tempered_posteriors[15:19, :])
#     training_p_wrapped = SoftmaxCompresser.compress_probability(modes=modes, probability=training_tempered_posteriors)
#     print("X")

training_sample_count = training_features.shape[0]
test_sample_count = test_features.shape[0]
db_rows = []

sess = tf.Session()
# p: The tempered posteriors, which have been squashed.
p = tf.placeholder(tf.float32, shape=(None, class_count))
# t: The squashed one hot labels
t = tf.placeholder(tf.float32, shape=(None, class_count))
features_tensor = tf.placeholder(tf.float32, shape=(None, features_dim))
soft_labels_cost_weight = tf.placeholder(tf.float32)
hard_labels_cost_weight = tf.placeholder(tf.float32)
l2_loss_weight = tf.placeholder(tf.float32)
keep_prob_tensor = tf.placeholder(tf.float32)
# Get new class count: Mode labels + Outliers. Init the new classifier hyperplanes.
hidden_weights = tf.Variable(
    tf.truncated_normal([features_dim, hidden_feature_dim],
                        stddev=0.1,
                        seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE),
    name="hidden_weights")
hidden_biases = tf.Variable(
    tf.constant(0.1, shape=[hidden_feature_dim], dtype=GlobalConstants.DATA_TYPE),
    name="hidden_biases")
softmax_weights = tf.Variable(
    tf.truncated_normal([hidden_feature_dim, class_count],
                        stddev=0.1,
                        seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE),
    name="softmax_weights")
softmax_biases = tf.Variable(
    tf.constant(0.1, shape=[class_count], dtype=GlobalConstants.DATA_TYPE),
    name="softmax_biases")

# NN
hidden_layer = tf.nn.relu(tf.matmul(features_tensor, hidden_weights) + hidden_biases)
hidden_layer_dropped = tf.nn.dropout(hidden_layer, keep_prob=keep_prob_tensor)
logits = tf.matmul(hidden_layer_dropped, softmax_weights) + softmax_biases

result_probs = tf.nn.softmax(logits)
# Term 1: Cross entropy between the soft labels and q
soft_loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=logits)
soft_loss = soft_labels_cost_weight * tf.reduce_mean(soft_loss_vec)
# Term 2: Cross entropy between the hard labels and q
hard_loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logits)
hard_loss = hard_labels_cost_weight * tf.reduce_mean(hard_loss_vec)
# Term 3: L2 loss for softmax weights
regularization_loss = tf.add_n([tf.nn.l2_loss(softmax_weights), tf.nn.l2_loss(hidden_weights)])
weight_l2 = l2_loss_weight * regularization_loss
# Total loss
total_loss = soft_loss + hard_loss + weight_l2
# Global step counter and Batch size
global_step = tf.Variable(name="global_step", initial_value=0, trainable=False)
batch_size = int(float(training_sample_count) * GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO)

# Calculate accuracy on the training set
training_accuracy_full = \
    SoftmaxCompresser.calculate_compressed_accuracy(posteriors=training_compressed_posteriors,
                                                    one_hot_labels=training_one_hot_labels)
# Calculate accuracy on the validation set
test_accuracy_full = \
    SoftmaxCompresser.calculate_compressed_accuracy(posteriors=test_compressed_posteriors,
                                                    one_hot_labels=test_one_hot_labels)
db_rows.append((run_id, -1, node_index, -1, -1, -1, -1, -1, -1, -1, -1, 0, training_accuracy_full))
db_rows.append((run_id, -1, node_index, -1, -1, -1, -1, -1, -1, -1, -1, 1, test_accuracy_full))
DbLogger.write_into_table(rows=db_rows, table=DbLogger.compressionTestsTable, col_count=13)

# temperature_list = [1.0]
# soft_loss_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
# hard_loss_weights = [1.0]
# l2_weights = [0.0, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
# learning_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# cross_validation_repeat_count = 10

temperature_list = [1.0]
soft_loss_weights = [0.0]
hard_loss_weights = [1.0]
# l2_weights = [0.0, 0.00001, 0.00005]
# l2_weights.extend([(i + 1) * 0.0001 for i in range(30)])
l2_weights = [0.0]
keep_probabilities = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
learning_rates = [0.00001, 0.000025, 0.00005,
                  0.0001, 0.00025, 0.0005,
                  0.001, 0.0025, 0.005,
                  0.01, 0.025, 0.05,
                  0.1, 0.25, 0.5]
cross_validation_repeat_count = 10

cartesian_product_soft_loss_changing = UtilityFuncs.get_cartesian_product(list_of_lists=[learning_rates,
                                                                                         temperature_list,
                                                                                         soft_loss_weights,
                                                                                         hard_loss_weights,
                                                                                         l2_weights,
                                                                                         keep_probabilities])
cartesian_product_hard_loss_changing = UtilityFuncs.get_cartesian_product(list_of_lists=[learning_rates,
                                                                                         temperature_list,
                                                                                         hard_loss_weights,
                                                                                         soft_loss_weights,
                                                                                         l2_weights,
                                                                                         keep_probabilities])
all_cartesian_products = []
all_cartesian_products.extend(cartesian_product_soft_loss_changing)
all_cartesian_products.extend(cartesian_product_hard_loss_changing)
all_cartesian_products = sorted(all_cartesian_products, key=lambda params_tuple: params_tuple[0])
duplicate_cartesians = []
for tpl in all_cartesian_products:
    duplicate_cartesians.extend(list(itertools.repeat(tpl, cross_validation_repeat_count)))
# Cross Validation
curr_lr = 0.0
learning_rate = None
trainer = None
p_dict = {}
# A new run for each tuple
for tpl in duplicate_cartesians:
    lr = tpl[0]
    temperature = tpl[1]
    soft_loss_weight = tpl[2]
    hard_loss_weight = tpl[3]
    l2_weight = tpl[4]
    keep = tpl[5]
    db_rows = []
    if curr_lr != lr:
        learning_rate = tf.train.exponential_decay(lr, global_step,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_DECAY,
                                                   staircase=True)
        trainer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, global_step=global_step)
        curr_lr = lr
    # Training sets
    training_indices = list(range(training_sample_count))
    shuffle(training_indices)
    random_indices = np.random.uniform(0, training_sample_count, batch_size).astype(int).tolist()

    training_t = training_one_hot_labels[training_indices]
    training_x = training_features[training_indices]
    training_logits = training_logits[training_indices]

    training_indices.extend(random_indices)

    training_t_wrapped = training_one_hot_labels[training_indices]
    training_x_wrapped = training_features[training_indices]
    training_logits_wrapped = training_logits[training_indices]
    # Produce p
    training_tempered_posteriors = SoftmaxCompresser.get_tempered_probabilities(logits=training_logits_wrapped,
                                                                                temperature=temperature)
    training_p_wrapped = SoftmaxCompresser.compress_probability(modes=modes, probability=training_tempered_posteriors)

    # Test sets
    test_indices = list(range(test_sample_count))
    test_t = test_one_hot_labels[test_indices]
    test_x = test_features[test_indices]
    # Init parameters
    all_variables = tf.global_variables()
    momentum_vars = [var for var in all_variables if "/Momentum" in var.name]
    vars_to_init = []
    vars_to_init.extend(momentum_vars)
    vars_to_init.extend([hidden_weights, hidden_biases, softmax_weights, softmax_biases, global_step])
    # Init variables
    init_op = tf.variables_initializer(vars_to_init)
    sess.run(init_op)
    # momentum_values_after_init = sess.run(momentum_vars)
    iteration = 0
    lr_last = lr
    for epoch_id in range(GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT):
        curr_index = 0
        while True:
            t_batch = training_t_wrapped[curr_index:curr_index + batch_size]
            p_batch = training_p_wrapped[curr_index:curr_index + batch_size]
            features_batch = training_x_wrapped[curr_index:curr_index + batch_size]
            feed_dict = {t: t_batch,
                         p: p_batch,
                         features_tensor: features_batch,
                         soft_labels_cost_weight: soft_loss_weight,
                         hard_labels_cost_weight: hard_loss_weight,
                         l2_loss_weight: l2_weight,
                         keep_prob_tensor: keep}
            run_ops = [trainer, learning_rate]
            results = sess.run(run_ops, feed_dict=feed_dict)
            # momentum_values = sess.run(momentum_vars)
            iteration += 1
            # print("Iteration:{0} Learning Rate:{1}".format(iteration, results[-1]))
            if results[-1] != lr_last:
                lr_last = results[-1]
                print("Iteration:{0} Learning Rate:{1}".format(iteration, lr_last))
            curr_index += batch_size
            if curr_index >= training_sample_count:
                is_last_epoch = epoch_id == GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT - 1
                # Evaluate on training set
                training_results = sess.run(
                    [result_probs],
                    feed_dict={t: training_t,
                               features_tensor: training_x,
                               l2_loss_weight: l2_weight,
                               keep_prob_tensor: 1.0})
                training_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                    posteriors=training_results[0], one_hot_labels=training_t)
                # Evaluate on test set
                test_results = sess.run(
                    [result_probs],
                    feed_dict={t: test_t,
                               features_tensor: test_x,
                               l2_loss_weight: l2_weight,
                               keep_prob_tensor: 1.0})
                test_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                    posteriors=test_results[0], one_hot_labels=test_t)
                # # Get resulting linear classifiers
                # hyperplane_weights = training_results[2]
                # hyperplane_biases = training_results[3]
                # print("Uncompressed Training Accuracy:{0}".format(training_accuracy_full))
                # print("Uncompressed Test Accuracy:{0}".format(test_accuracy_full))
                # print("Compressed Training Accuracy:{0}".format(training_accuracy))
                # print("Compressed Test Accuracy:{0}".format(test_accuracy))
                if GlobalConstants.USE_SOFTMAX_DISTILLATION_VERBOSE:
                    db_rows.append((run_id, iteration, node_index, temperature, soft_loss_weight, hard_loss_weight,
                                    l2_weight, lr, keep, GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT,
                                    GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT, 0, training_accuracy))
                    db_rows.append((run_id, iteration, node_index, temperature, soft_loss_weight, hard_loss_weight,
                                    l2_weight, lr, keep, GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT,
                                    GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT, 1, test_accuracy))
                # if is_last_epoch:
                #     final_softmax_weights = hyperplane_weights
                #     final_softmax_biases = hyperplane_biases
                #     final_training_accuracy = training_accuracy
                #     final_test_accuracy = test_accuracy
                #     return final_softmax_weights, final_softmax_biases, final_training_accuracy, final_test_accuracy
                break
        if epoch_id == GlobalConstants.SOFTMAX_DISTILLATION_EPOCH_COUNT - 1:
            print("Training Accuracy:{0}".format(training_accuracy))
            print("Test Accuracy:{0}".format(test_accuracy))
    DbLogger.write_into_table(rows=db_rows, table=DbLogger.compressionTestsTable, col_count=13)
