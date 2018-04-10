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
