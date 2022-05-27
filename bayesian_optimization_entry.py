import os.path

import tensorflow as tf

from auxillary.db_logger import DbLogger
from tf_2_cign.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer
from tf_2_cign.bayesian_optimizers.fmnist_gumbel_softmax_optimizer_with_decay_rate import \
    FmnistGumbelSoftmaxOptimizerWithDecayRate
from tf_2_cign.bayesian_optimizers.fmnist_gumbel_softmax_optimizer_with_lr_decay_rate \
    import FmnistGumbelSoftmaxOptimizerWithLrDecayRate
from tf_2_cign.bayesian_optimizers.fmnist_gumbel_softmax_vanilla_optimizer import FmnistGumbelSoftmaxVanilla
from tf_2_cign.bayesian_optimizers.fmnist_random_routing_optimizer import FmnistRandomRoutingOptimizer
from tf_2_cign.entry_points.fashion_cigt_bo_hyperparameter_search import optimize_with_bayesian_optimization

# from auxillary.db_logger import DbLogger
# from auxillary.general_utility_funcs import UtilityFuncs
# Hyper-parameters


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    DbLogger.log_db_path = DbLogger.blackshark_desktop
    bayesian_optimizer = FmnistGumbelSoftmaxVanilla(init_points=100, n_iter=300, xi=0.01)
    bayesian_optimizer.fit(log_file_root_path=os.path.split(os.path.abspath(__file__))[0],
                           log_file_name="bo_gumbel_softmax_mean_z_vanilla")

