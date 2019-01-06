from enum import Enum

import tensorflow as tf

from auxillary.constants import DatasetTypes
from auxillary.parameters import DecayingParameter, DiscreteParameter, DecayingParameterV2
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator


class GradientType(Enum):
    mixture_of_experts_unbiased = 0
    mixture_of_experts_biased = 1
    parallel_dnns_unbiased = 2
    parallel_dnns_biased = 3


class AccuracyCalcType(Enum):
    regular = 0
    route_correction = 1
    with_residue_network = 2
    multi_path = 3


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
    BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 10000
    CURR_BATCH_SIZE = None
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
    INITIAL_LR = 0.1
    DECAY_STEP = 15000
    DECAY_RATE = 0.5  # INITIAL_LR/EPOCH_COUNT
    # LEARNING_RATE_CALCULATOR = DecayingParameterV2(name="lr_calculator", value=INITIAL_LR,
    #                                                decay=DECAY_RATE)
    # LEARNING_RATE_CALCULATOR = DecayingParameter(name="lr_calculator", value=INITIAL_LR, decay=DECAY_RATE,
    #                                              decay_period=DECAY_STEP)
    LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator", value=INITIAL_LR,
                                                 schedule=[(40000, 0.01),
                                                           (60000, 0.001)])

    TREE_DEGREE = 2
    MOMENTUM_DECAY = 0.9
    BATCH_NORM_DECAY = 0.9
    # PROBABILITY_THRESHOLD = DecayingParameter(name="ProbThreshold", value=1.0 / float(TREE_DEGREE), decay=0.999,
    #                                           decay_period=1, min_limit=0.0)
    SOFTMAX_DECAY_INITIAL = 25.0
    SOFTMAX_DECAY_COEFFICIENT = 0.9999
    SOFTMAX_DECAY_PERIOD = 2
    SOFTMAX_DECAY_MIN_LIMIT = 1.0
    SOFTMAX_TEST_TEMPERATURE = 50.0
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
    USE_SAMPLE_HASHING = False
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
    USE_UNIFIED_BATCH_NORM = True
    USE_TRAINABLE_PARAMS_WITH_BATCH_NORM = True
    USE_DECISION_REGULARIZER = True
    DECISION_LOSS_COEFFICIENT = 1.0
    SAVE_CONFUSION_MATRICES = False
    GRADIENT_TYPE = GradientType.parallel_dnns_unbiased
    INFO_GAIN_LOG_EPSILON = 1e-30
    SUMMARY_PERIOD = 100000000000
    # CLASS WEIGHTING
    USE_CLASS_WEIGHTING = False
    CLASS_WEIGHT_RUNNING_AVERAGE = 0.9
    LABEL_EPSILON = 0.1
    # Fashion Mnist
    # Baseline
    FASHION_NUM_FILTERS_1 = 32
    FASHION_NUM_FILTERS_2 = 64
    FASHION_NUM_FILTERS_3 = 64
    FASHION_FILTERS_1_SIZE = 5
    FASHION_FILTERS_2_SIZE = 5
    FASHION_FILTERS_3_SIZE = 1
    FASHION_FC_1 = 128
    FASHION_FC_2 = 64
    BASELINE_ENSEMBLE_COUNT = 1
    # Conditional [2 2] Tree
    FASHION_F_NUM_FILTERS_1 = 32
    FASHION_F_NUM_FILTERS_2 = 32
    FASHION_F_NUM_FILTERS_3 = 32
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
    FASHION_NO_H_FROM_F_UNITS_1 = 128
    FASHION_NO_H_FROM_F_UNITS_2 = 128

    # Residue Network
    RESIDUE_LOSS_COEFFICIENT = 0.0
    RESIDE_AFFECTS_WHOLE_NETWORK = True
    FASHION_F_RESIDUE = 128
    FASHION_F_RESIDUE_LAYER_COUNT = 1
    FASHION_F_RESIDUE_USE_DROPOUT = False

    # Resnet Baseline Params
    RESNET_HYPERPARAMS = ResnetGenerator.ResnetHParams(num_residual_units=54, use_bottleneck=True,
                                                       num_of_features_per_block=[16, 64, 128, 256],
                                                       first_conv_filter_size=3, relu_leakiness=0.1,
                                                       strides=[1, 2, 2], activate_before_residual=[True, False, False])

    # MultiPath Evaluation Schedules
    # MULTIPATH_SCHEDULES = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    # MULTIPATH_SCHEDULES.extend([i*0.001 for i in range(50)])
    MULTIPATH_SCHEDULES = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15,
                           0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.01,
                           0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.0025, 0.002, 0.001,
                           0.0005, 0.00025, 0.0001,
                           0.00005, 0.000025, 0.00001,
                           0.000005, 0.0000025, 0.000001,
                           0.0000005, 0.00000025, 0.0000001,
                           0.00000005, 0.000000025, 0.00000001,
                           0.000000005, 0.0000000025, 0.000000001,
                           0.0000000005, 0.00000000025, 0.0000000001,
                           0.00000000005, 0.000000000025, 0.00000000001,
                           0.000000000005, 0.0000000000025, 0.000000000001,
                           0.0000000000005, 0.00000000000025, 0.0000000000001,
                           0.00000000000005, 0.000000000000025, 0.00000000000001,
                           0.000000000000005, 0.0000000000000025, 0.000000000000001,
                           0.0000000000000005, 0.00000000000000025, 0.0000000000000001,
                           0.00000000000000005, 0.000000000000000025, 0.00000000000000001,
                           0.000000000000000005, 0.0000000000000000025, 0.000000000000000001,
                           0.0000000000000000005, 0.00000000000000000025, 0.0000000000000000001,
                           0.00000000000000000005, 0.000000000000000000025, 0.00000000000000000001,
                           0.0]
    # Idea
    # SUMMARY_DIR = "C://Users//ufuk.bicici//Desktop//tf//phd_work//simple_tf"
    # Home
    SUMMARY_DIR = "C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf"
    # TRAIN
    TRAIN_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    TRAIN_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(None,))
    TRAIN_INDEX_TENSOR = tf.placeholder(tf.int64, shape=(None,))
    TRAIN_ONE_HOT_LABELS = tf.placeholder(dtype=DATA_TYPE, shape=(None, NUM_LABELS))

    BATCH_SIZES_DICT = {DatasetTypes.training: BATCH_SIZE,
                        DatasetTypes.test: EVAL_BATCH_SIZE,
                        DatasetTypes.validation: EVAL_BATCH_SIZE}
