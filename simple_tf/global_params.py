from enum import Enum

import tensorflow as tf

from auxillary.parameters import DecayingParameter


class GradientType(Enum):
    mixture_of_experts_unbiased = 0
    mixture_of_experts_biased = 1
    parallel_dnns_unbiased = 2
    parallel_dnns_biased = 3


class GlobalConstants:
    EPOCH_COUNT = 6000
    BATCH_SIZE = 5000
    EVAL_BATCH_SIZE = 50000
    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NO_FILTERS_1 = 20
    NO_FILTERS_2 = 13  # 10
    NO_HIDDEN = 10  # 30
    NUM_LABELS = 10
    WEIGHT_DECAY_COEFFICIENT = 0.0
    INITIAL_LR = 0.025
    DECAY_STEP = 20000
    DECAY_RATE = 0.5
    TREE_DEGREE = 2
    MOMENTUM_DECAY = 0.9
    PROBABILITY_THRESHOLD = DecayingParameter(name="ProbThreshold", value=1.0 / float(TREE_DEGREE), decay=0.999,
                                              decay_period=1, min_limit=0.0)
    DATA_TYPE = tf.float32
    SEED = None
    USE_CPU = False
    USE_CPU_MASKING = False
    USE_EMPTY_NODE_CRASH_PREVENTION = False
    USE_RANDOM_PARAMETERS = True
    USE_PROBABILITY_THRESHOLD = False
    USE_INFO_GAIN_DECISION = True
    USE_CONCAT_TRICK = False
    USE_BATCH_NORM_BEFORE_BRANCHING = False
    USE_TRAINABLE_PARAMS_WITH_BATCH_NORM = False
    DECISION_LOSS_COEFFICIENT = 1.0
    SAVE_CONFUSION_MATRICES = False
    GRADIENT_TYPE = GradientType.mixture_of_experts_biased
    INFO_GAIN_LOG_EPSILON = 1e-30
    SUMMARY_PERIOD = 100000000000
    TREE_DEGREE_LIST = [3, 2]
    # Idea
    # SUMMARY_DIR = "C://Users//ufuk.bicici//Desktop//tf//phd_work//simple_tf"
    # Home
    SUMMARY_DIR = "C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf"
    # TRAIN
    TRAIN_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    TRAIN_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    TRAIN_ONE_HOT_LABELS = tf.placeholder(dtype=DATA_TYPE, shape=(BATCH_SIZE, NUM_LABELS))
    # TEST
    # TEST_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    # TEST_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(EVAL_BATCH_SIZE,))
    # TEST_ONE_HOT_LABELS = tf.placeholder(dtype=DATA_TYPE, shape=(EVAL_BATCH_SIZE, NUM_LABELS))
