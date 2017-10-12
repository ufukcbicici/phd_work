import tensorflow as tf


class GlobalConstants:
    EPOCH_COUNT = 1000
    BATCH_SIZE = 1000
    EVAL_BATCH_SIZE = 50000
    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NO_FILTERS_1 = 20
    NO_FILTERS_2 = 25
    NO_HIDDEN = 125
    NUM_LABELS = 10
    WEIGHT_DECAY_COEFFICIENT = 0.0
    INITIAL_LR = 0.01
    DECAY_STEP = 10000
    DECAY_RATE = 0.5
    TREE_DEGREE = 2
    MOMENTUM_DECAY = 0.9
    DATA_TYPE = tf.float32
    SEED = None
    USE_CPU = False
    USE_CPU_MASKING = False
    USE_RANDOM_PARAMETERS = True
    TRAIN_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    TRAIN_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    TEST_DATA_TENSOR = tf.placeholder(DATA_TYPE, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    TEST_LABEL_TENSOR = tf.placeholder(tf.int64, shape=(EVAL_BATCH_SIZE,))
