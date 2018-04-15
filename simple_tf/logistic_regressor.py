from algorithms.softmax_compresser import SoftmaxCompresser
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import tensorflow as tf
import itertools
from random import shuffle
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

class_count = 4
features_dim = 64
node_index = 3
rund_id = 0

node_3_features_dict = UtilityFuncs.load_npz(file_name="npz_node_3_final_features")

training_features = node_3_features_dict["training_features"]
training_one_hot_labels = node_3_features_dict["training_one_hot_labels"]
training_compressed_posteriors = node_3_features_dict["training_compressed_posteriors"]

test_features = node_3_features_dict["test_features"]
test_one_hot_labels = node_3_features_dict["test_one_hot_labels"]
test_compressed_posteriors = node_3_features_dict["test_compressed_posteriors"]

data_scaler = RobustScaler()
normalized_training_features = data_scaler.fit_transform(training_features)
normalized_test_features = data_scaler.transform(test_features)

training_features = normalized_training_features
test_features = normalized_test_features

training_sample_count = training_features.shape[0]
test_sample_count = test_features.shape[0]
kv_rows = []

sess = tf.Session()
# t: The squashed one hot labels
t = tf.placeholder(tf.float32, shape=(None, class_count))
features_tensor = tf.placeholder(tf.float32, shape=(None, features_dim))
l2_loss_weight = tf.placeholder(tf.float32)
# Get new class count: Mode labels + Outliers. Init the new classifier hyperplanes.
softmax_weights = tf.Variable(
    tf.truncated_normal([features_dim, class_count],
                        stddev=0.1,
                        seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE),
    name="softmax_weights")
softmax_biases = tf.Variable(
    tf.constant(0.1, shape=[class_count], dtype=GlobalConstants.DATA_TYPE),
    name="softmax_biases")
# Compressed softmax probabilities
logits = tf.matmul(features_tensor, softmax_weights) + softmax_biases
result_probs = tf.nn.softmax(logits)
# Term 2: Cross entropy between the hard labels and q
hard_loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logits)
hard_loss = tf.reduce_mean(hard_loss_vec)
# Term 3: L2 loss for softmax weights
weight_l2 = l2_loss_weight * tf.nn.l2_loss(softmax_weights)

total_loss = hard_loss + weight_l2
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
kv_rows.append((rund_id, -1,
                "Leaf:{0} Training Accuracy Full".format(node_index), training_accuracy_full))
kv_rows.append((rund_id, -1,
                "Leaf:{0} Test Accuracy Full".format(node_index), test_accuracy_full))
DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)

# temperature_list = [1.0]
# soft_loss_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
# hard_loss_weights = [1.0]
# l2_weights = [0.0, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
# learning_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# cross_validation_repeat_count = 10

temperature_list = [1.0]
soft_loss_weights = [0.0]  # [0.0, 0.25, 0.5, 0.75, 1.0]
hard_loss_weights = [1.0]
l2_weights = [0.0, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002]
learning_rates = [0.0025, 0.005,
                  0.01, 0.025, 0.05,
                  0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# [0.00001, 0.000025, 0.00005,
#                  0.0001, 0.00025, 0.0005,
#                  0.001,

cross_validation_repeat_count = 10

cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[learning_rates,
                                                                      temperature_list, soft_loss_weights,
                                                                      hard_loss_weights,
                                                                      l2_weights])

duplicate_cartesians = []
for tpl in cartesian_product:
    duplicate_cartesians.extend(list(itertools.repeat(tpl, cross_validation_repeat_count)))

# Cross Validation
curr_lr = 0.0
learning_rate = None
trainer = None
# A new run for each tuple
for tpl in duplicate_cartesians:
    lr = tpl[0]
    temperature = tpl[1]
    soft_loss_weight = tpl[2]
    hard_loss_weight = tpl[3]
    l2_weight = tpl[4]
    kv_rows = []
    if curr_lr != lr:
        learning_rate = tf.train.exponential_decay(lr, global_step,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_DECAY,
                                                   staircase=True)
        trainer = tf.train.MomentumOptimizer(learning_rate, 0.0).minimize(total_loss, global_step=global_step)
        curr_lr = lr
    # Training sets
    training_indices = list(range(training_sample_count))
    shuffle(training_indices)
    random_indices = np.random.uniform(0, training_sample_count, batch_size).astype(int).tolist()
    training_t = training_one_hot_labels[training_indices]
    training_x = training_features[training_indices]
    training_indices.extend(random_indices)
    training_t_wrapped = training_one_hot_labels[training_indices]
    training_x_wrapped = training_features[training_indices]
    # Test sets
    test_indices = list(range(test_sample_count))
    test_t = test_one_hot_labels[test_indices]
    test_x = test_features[test_indices]
    # Init parameters
    all_variables = tf.global_variables()
    momentum_vars = [var for var in all_variables if "/Momentum" in var.name]
    vars_to_init = []
    vars_to_init.extend(momentum_vars)
    vars_to_init.extend([softmax_weights, softmax_biases, global_step])
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
            features_batch = training_x_wrapped[curr_index:curr_index + batch_size]
            feed_dict = {t: t_batch,
                         features_tensor: features_batch,
                         l2_loss_weight: l2_weight}
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
                    [result_probs, total_loss, softmax_weights, softmax_biases],
                    feed_dict={t: training_t,
                               features_tensor: training_x,
                               l2_loss_weight: l2_weight})
                training_accuracy = SoftmaxCompresser.calculate_compressed_accuracy(
                    posteriors=training_results[0], one_hot_labels=training_t)
                # Evaluate on test set
                test_results = sess.run(
                    [result_probs, total_loss, softmax_weights, softmax_biases],
                    feed_dict={t: test_t,
                               features_tensor: test_x,
                               l2_loss_weight: l2_weight})
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
                    kv_table_key = "Leaf:{0} T:{1} slW:{2} hlW:{3} l2W:{4} lr:{5}".format(node_index,
                                                                                          temperature,
                                                                                          soft_loss_weight,
                                                                                          hard_loss_weight,
                                                                                          l2_weight, lr
                                                                                          )
                    kv_rows.append((rund_id, iteration, "Training Accuracy {0}".format(kv_table_key),
                                    training_accuracy))
                    kv_rows.append((rund_id, iteration, "Test Accuracy {0}".format(kv_table_key),
                                    test_accuracy))
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
    DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
