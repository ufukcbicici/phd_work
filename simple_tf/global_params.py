from enum import Enum

import tensorflow as tf

from auxillary.parameters import DecayingParameter, DiscreteParameter, DecayingParameterV2


class GradientType(Enum):
    mixture_of_experts_unbiased = 0
    mixture_of_experts_biased = 1
    parallel_dnns_unbiased = 2
    parallel_dnns_biased = 3


class GlobalConstants:
    TOTAL_EPOCH_COUNT = 100
    EPOCH_COUNT = 100
    EPOCH_REPORT_PERIOD = 1
    BATCH_SIZE = 125
    EVAL_BATCH_SIZE = 10000
    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    # TREE_DEGREE_LIST = [3, 2]
    # NO_FILTERS_1 = 20
    # NO_FILTERS_2 = 13  # 10
    # NO_HIDDEN = 10  # 30
    TREE_DEGREE_LIST = [2, 2]
    NO_FILTERS_1 = 20
    NO_FILTERS_2 = 15  # 10
    NO_HIDDEN = 25  # 30
    NUM_LABELS = 10
    WEIGHT_DECAY_COEFFICIENT = 0.0
    DECISION_WEIGHT_DECAY_COEFFICIENT = 0.0
    INITIAL_LR = 0.01
    DECAY_STEP = 15000
    DECAY_RATE = 0.5 # INITIAL_LR/EPOCH_COUNT
    # LEARNING_RATE_CALCULATOR = DecayingParameterV2(name="lr_calculator", value=INITIAL_LR,
    #                                                decay=DECAY_RATE)
    LEARNING_RATE_CALCULATOR = DecayingParameter(name="lr_calculator", value=INITIAL_LR, decay=DECAY_RATE,
                                                 decay_period=DECAY_STEP)

    TREE_DEGREE = 2
    MOMENTUM_DECAY = 0.9
    BATCH_NORM_DECAY = 0.9
    # PROBABILITY_THRESHOLD = DecayingParameter(name="ProbThreshold", value=1.0 / float(TREE_DEGREE), decay=0.999,
    #                                           decay_period=1, min_limit=0.0)
    SOFTMAX_DECAY_INITIAL = 1.0
    SOFTMAX_DECAY_COEFFICIENT = 1.0
    SOFTMAX_DECAY_PERIOD = 1
    SOFTMAX_DECAY_MIN_LIMIT = 1.0
    DROPOUT_INITIAL_PROB = 0.75
    DROPOUT_SCHEDULE = [(15000, 0.5), (30000, 0.25), (45000, 0.125)]
    CLASSIFICATION_DROPOUT_PROB = 0.5
    # INFO_GAIN_BALANCE_COEFFICIENT = 1.0
    PERCENTILE_THRESHOLD = 0.95
    ROUTE_CORRECTION_PERIOD = 5000
    USE_CONVOLUTIONAL_H_PIPELINE = True
    NO_H_FILTERS_1 = 5
    NO_H_FC_UNITS_1 = 20
    NO_H_FILTERS_2 = 5
    NO_H_FC_UNITS_2 = 20
    NO_H_FROM_F_UNITS_1 = 20
    NO_H_FROM_F_UNITS_2 = 20
    DATA_TYPE = tf.float32
    SEED = None
    USE_VERBOSE = False
    USE_CPU = False
    USE_CPU_MASKING = False
    USE_EMPTY_NODE_CRASH_PREVENTION = False
    USE_RANDOM_PARAMETERS = True
    USE_PROBABILITY_THRESHOLD = False
    USE_ADAPTIVE_WEIGHT_DECAY = False
    ADAPTIVE_WEIGHT_DECAY_MIXING_RATE = 0.9
    USE_DROPOUT_FOR_DECISION = False
    USE_REPARAMETRIZATION_TRICK = False
    USE_DROPOUT_FOR_CLASSIFICATION = False
    INFO_GAIN_BALANCE_COEFFICIENT = 1.0
    USE_INFO_GAIN_DECISION = True
    USE_DECISION_AUGMENTATION = False
    USE_CONCAT_TRICK = False
    USE_BATCH_NORM_BEFORE_BRANCHING = True
    USE_TRAINABLE_PARAMS_WITH_BATCH_NORM = True
    USE_DECISION_REGULARIZER = True
    DECISION_LOSS_COEFFICIENT = 1.0
    RESIDUE_LOSS_COEFFICIENT = 0.0
    SAVE_CONFUSION_MATRICES = True
    GRADIENT_TYPE = GradientType.mixture_of_experts_biased
    INFO_GAIN_LOG_EPSILON = 1e-30
    SUMMARY_PERIOD = 100000000000
    # Fashion Mnist
    # Baseline
    FASHION_NUM_FILTERS_1 = 32
    FASHION_NUM_FILTERS_2 = 64
    FASHION_NUM_FILTERS_3 = 128
    FASHION_FILTERS_1_SIZE = 5
    FASHION_FILTERS_2_SIZE = 5
    FASHION_FILTERS_3_SIZE = 1
    FASHION_FC_1 = 1024
    FASHION_FC_2 = 512
    # Conditional [2 2] Tree
    FASHION_F_NUM_FILTERS_1 = 32
    FASHION_F_NUM_FILTERS_2 = 64
    FASHION_F_NUM_FILTERS_3 = 64
    FASHION_H_NUM_FILTERS_1 = 10
    FASHION_H_NUM_FILTERS_2 = 20
    FASHION_H_NUM_FILTERS_3 = 40
    FASHION_H_FILTERS_1_SIZE = 5
    FASHION_H_FILTERS_2_SIZE = 5
    FASHION_H_FILTERS_3_SIZE = 1
    FASHION_H_FC_1 = 30
    FASHION_H_FC_2 = 30
    FASHION_F_FC_1 = 192
    FASHION_F_FC_2 = 96
    FASHION_F_RESIDUE = 32


    # Idea
    # SUMMARY_DIR = "C://Users//ufuk.bicici//Desktop//tf//phd_work//simple_tf"
    # Home
    SUMMARY_DIR = "C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf"
    # TRAIN
    TRAIN_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    TRAIN_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(None,))
    TRAIN_INDEX_TENSOR = tf.placeholder(tf.int64, shape=(None,))
    TRAIN_ONE_HOT_LABELS = tf.placeholder(dtype=DATA_TYPE, shape=(None, NUM_LABELS))
    # TEST
    # TEST_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    # TEST_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(EVAL_BATCH_SIZE,))
    # TEST_ONE_HOT_LABELS = tf.placeholder(dtype=DATA_TYPE, shape=(EVAL_BATCH_SIZE, NUM_LABELS))
