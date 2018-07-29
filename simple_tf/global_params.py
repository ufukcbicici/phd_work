from enum import Enum

import tensorflow as tf

from auxillary.parameters import DecayingParameter, DiscreteParameter, DecayingParameterV2


class GradientType(Enum):
    mixture_of_experts_unbiased = 0
    mixture_of_experts_biased = 1
    parallel_dnns_unbiased = 2
    parallel_dnns_biased = 3


class AccuracyCalcType(Enum):
    regular = 0
    route_correction = 1
    with_residue_network = 2


class SoftmaxCompressionStrategy(Enum):
    random_start = 0
    fit_logistic_layer = 1
    fit_svm_layer = 2


class ModeComputationStrategy(Enum):
    percentile = 0
    max_num_of_classes = 1


class ModeTrackingStrategy(Enum):
    wait_for_convergence = 0
    wait_for_fixed_epochs = 1


class GlobalConstants:
    TOTAL_EPOCH_COUNT = 100
    EPOCH_COUNT = 100
    EPOCH_REPORT_PERIOD = 1
    BATCH_SIZE = 125
    EVAL_BATCH_SIZE = 10000
    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    USE_FAST_TREE_MODE = True
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
    # LEARNING_RATE_CALCULATOR = DecayingParameter(name="lr_calculator", value=INITIAL_LR, decay=DECAY_RATE,
    #                                              decay_period=DECAY_STEP)
    LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator", value=INITIAL_LR,
                                                 schedule=[(15000, 0.01),
                                                           (30000, 0.005),
                                                           (40000, 0.0005),
                                                           (64000, 0.00025)])

    TREE_DEGREE = 2
    MOMENTUM_DECAY = 0.9
    BATCH_NORM_DECAY = 0.9
    # PROBABILITY_THRESHOLD = DecayingParameter(name="ProbThreshold", value=1.0 / float(TREE_DEGREE), decay=0.999,
    #                                           decay_period=1, min_limit=0.0)
    SOFTMAX_DECAY_INITIAL = 25.0
    SOFTMAX_DECAY_COEFFICIENT = 0.9999
    SOFTMAX_DECAY_PERIOD = 2
    SOFTMAX_DECAY_MIN_LIMIT = 1.0
    DROPOUT_INITIAL_PROB = 0.75
    DROPOUT_SCHEDULE = [(15000, 0.5), (30000, 0.25), (45000, 0.125)]
    CLASSIFICATION_DROPOUT_PROB = 0.5
    DECISION_DROPOUT_PROB = 0.35
    # INFO_GAIN_BALANCE_COEFFICIENT = 1.0

    # Softmax Compression Parameters
    PERCENTILE_THRESHOLD = 0.85
    MAX_MODE_CLASSES = 5
    MODE_COMPUTATION_STRATEGY = ModeComputationStrategy.percentile
    MODE_TRACKING_STRATEGY = ModeTrackingStrategy.wait_for_convergence
    CONSTRAIN_WITH_COMPRESSION_LABEL_COUNT = False
    COMPRESSION_EPOCH = 10
    MODE_WAIT_EPOCHS = 25
    SOFTMAX_DISTILLATION_INITIAL_LR = 0.01
    SOFTMAX_DISTILLATION_DECAY = 0.5
    SOFTMAX_DISTILLATION_BATCH_SIZE = 1000

    # SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO = 1.0
    # SOFTMAX_DISTILLATION_STEP_COUNT = 500
    # SOFTMAX_DISTILLATION_EPOCH_COUNT = 2000

    # SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO = (1.0 / 480.0)
    # SOFTMAX_DISTILLATION_STEP_COUNT = 6000
    # SOFTMAX_DISTILLATION_EPOCH_COUNT = 200

    SOFTMAX_COMPRESSION_STRATEGY = SoftmaxCompressionStrategy.fit_svm_layer
    SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO = 0.05
    SOFTMAX_DISTILLATION_STEP_COUNT = 75 * int(1.0 / SOFTMAX_DISTILLATION_BATCH_SIZE_RATIO)
    SOFTMAX_DISTILLATION_EPOCH_COUNT = 300
    SOFTMAX_DISTILLATION_VERBOSE = False

    SOFTMAX_DISTILLATION_CROSS_VALIDATION_COUNT = 3
    USE_SOFTMAX_DISTILLATION = False
    SOFTMAX_DISTILLATION_CPU_COUNT = 8
    SOFTMAX_DISTILLATION_GRADIENT_TYPE = GradientType.parallel_dnns_unbiased
    # Softmax Compression Parameters

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
    USE_SOFTMAX_DISTILLATION_VERBOSE = True
    USE_CPU = False
    USE_CPU_MASKING = False
    USE_EMPTY_NODE_CRASH_PREVENTION = False
    USE_RANDOM_PARAMETERS = True
    USE_PROBABILITY_THRESHOLD = True
    USE_ADAPTIVE_WEIGHT_DECAY = False
    ADAPTIVE_WEIGHT_DECAY_MIXING_RATE = 0.9
    USE_DROPOUT_FOR_DECISION = False
    USE_EFFECTIVE_SAMPLE_COUNTS = True
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
    SAVE_CONFUSION_MATRICES = False
    GRADIENT_TYPE = GradientType.parallel_dnns_unbiased
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
    FASHION_H_FC_1 = 32
    FASHION_H_FC_2 = 32
    FASHION_F_FC_1 = 128
    FASHION_F_FC_2 = 64
    FASHION_NO_H_FROM_F_UNITS_1 = 16
    FASHION_NO_H_FROM_F_UNITS_2 = 16

    # Residue Network
    RESIDUE_LOSS_COEFFICIENT = 1.0
    RESIDE_AFFECTS_WHOLE_NETWORK = True
    FASHION_F_RESIDUE = 128
    FASHION_F_RESIDUE_LAYER_COUNT = 1
    FASHION_F_RESIDUE_USE_DROPOUT = False

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
