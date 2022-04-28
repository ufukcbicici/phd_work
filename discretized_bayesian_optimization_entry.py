import tensorflow as tf

from tf_2_cign.entry_points.fashion_cigt_bo_hyperparameter_search import *

# from auxillary.db_logger import DbLogger
# from auxillary.general_utility_funcs import UtilityFuncs
# Hyper-parameters

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    optimize_with_discretized_bayesian_optimization()
