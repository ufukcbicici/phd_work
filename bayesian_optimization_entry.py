import os.path

import tensorflow as tf

from auxillary.db_logger import DbLogger
from tf_2_cign.cigt.bayesian_optimizers.fashion_mnist_lenet_cross_entropy_search import \
    FashionMnistLenetCrossEntropySearch
from tf_2_cign.cigt.bayesian_optimizers.fashion_mnist_lenet_threshold_optimizer import FashionMnistLenetThresholdOptimizer

# from auxillary.db_logger import DbLogger
# from auxillary.general_utility_funcs import UtilityFuncs
# Hyper-parameters


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # DbLogger.log_db_path = DbLogger.blackshark_desktop
    # bayesian_optimizer = FmnistGumbelSoftmaxOnlyDropoutOptimizer(init_points=100, n_iter=300, xi=0.01,
    #                                                              ig_balance_coeff=3.7233209862205525,
    #                                                              d_loss_coeff=0.7564802988471849)
    # bayesian_optimizer.fit(log_file_root_path=os.path.split(os.path.abspath(__file__))[0],
    #                        log_file_name="bo_gumbel_softmax_mean_z_only_dropout")

    DbLogger.log_db_path = DbLogger.blackshark_desktop
    # bayesian_optimizer = FmnistGumbelSoftmaxVanillaOptimizer(init_points=100, n_iter=300, xi=0.01)
    # bayesian_optimizer.fit(log_file_root_path=os.path.split(os.path.abspath(__file__))[0],
    #                        log_file_name="bo_gumbel_softmax_mean_z_vanilla")
    # bayesian_optimizer = FashionMnistLenetThresholdOptimizer(
    #     init_points=300, n_iter=700, xi=0.01, model_id=514, val_ratio=0.25, accuracy_mac_balance_coeff=1.0)
    # bayesian_optimizer.apply_brute_force_solution(indices=None)
    # bayesian_optimizer.fit(log_file_root_path=os.path.split(os.path.abspath(__file__))[0],
    #                        log_file_name="fmnist_multipath_optimization")

    cross_entropy_optimizer = FashionMnistLenetCrossEntropySearch(init_points=300, n_iter=700,
                                                                  xi=0.01, model_id=424,
                                                                  val_ratio=0.25, accuracy_mac_balance_coeff=1.0)
    cross_entropy_optimizer.run()
