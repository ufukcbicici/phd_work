from algorithms.softmax_compresser import SoftmaxCompresser
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import tensorflow as tf
import itertools
from random import shuffle

from simple_tf.global_params import GlobalConstants

class_count = 3
features_dim = 64

node_5_features_dict = UtilityFuncs.load_npz(file_name="npz_node_5_final_features")
training_features = node_5_features_dict["training_features"]
training_one_hot_labels = node_5_features_dict["training_one_hot_labels"]
test_features = node_5_features_dict["test_features"]
test_one_hot_labels = node_5_features_dict["test_one_hot_labels"]
training_sample_count = training_features.shape[0]
test_sample_count = test_features.shape[0]

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
# Term 2: Cross entropy between the hard labels and q
hard_loss_vec = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logits)
hard_loss = tf.reduce_mean(hard_loss_vec)
# Term 3: L2 loss for softmax weights
weight_l2 = l2_loss_weight * tf.nn.l2_loss(softmax_weights)

total_loss = hard_loss + weight_l2
global_step = tf.Variable(name="global_step", initial_value=0, trainable=False)
batch_size = int(float(training_sample_count) * GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO)

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
        learning_rate = tf.train.exponential_decay(lr, global_step,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_STEP_COUNT,
                                                   GlobalConstants.SOFTMAX_DISTILLATION_DECAY,
                                                   staircase=True)
        trainer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, global_step=global_step)
        curr_lr = lr
    # Training sets
    training_indices = list(range(training_sample_count))
    shuffle(training_indices)
    random_indices = np.random.uniform(0, training_sample_count,
                                       GlobalConstants.SOFTMAX_DISTILLATION_BATCH_SIZE).astype(int).tolist()
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
    vars_to_init = [var for var in all_variables if "/Momentum" in var.name]
    vars_to_init.extend([softmax_weights, softmax_biases, global_step])
    # Init variables
    init_op = tf.variables_initializer(vars_to_init)
    sess.run(init_op)
    iteration = 0
