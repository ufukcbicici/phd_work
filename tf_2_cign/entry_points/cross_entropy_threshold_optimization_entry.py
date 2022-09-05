import os.path

import tensorflow as tf
import json
import requests

from auxillary.db_logger import DbLogger
from tf_2_cign.cigt.bayesian_optimizers.fashion_mnist_lenet_cross_entropy_search import \
    FashionMnistLenetCrossEntropySearch
from tf_2_cign.cigt.bayesian_optimizers.fashion_mnist_lenet_sigmoid_norm_ce_search import \
    FashionMnistLenetSigmoidNormCeSearh
from tf_2_cign.cigt.bayesian_optimizers.fashion_mnist_lenet_threshold_optimizer import \
    FashionMnistLenetThresholdOptimizer

# from auxillary.db_logger import DbLogger
# from auxillary.general_utility_funcs import UtilityFuncs
# Hyper-parameters
from tf_2_cign.cigt.cross_entropy_optimizers.categorical_ce_threshold_optimizer import CategoricalCeThresholdOptimizer
from tf_2_cign.cigt.cross_entropy_optimizers.cross_entropy_threshold_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.cross_entropy_optimizers.high_entropy_threshold_optimizer import HighEntropyThresholdOptimizer
from tf_2_cign.cigt.cross_entropy_optimizers.sigmoid_gmm_ce_threshold_optimizer import SigmoidGmmCeThresholdOptimizer
from tf_2_cign.cigt.model_loaders.fmnist_lenet_pretrained_model_loader import FmnistLenetPretrainedModelLoader

DbLogger.log_db_path = DbLogger.blackshark_laptop
output_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..", "tf_2_cign",
                           "cigt", "image_outputs")

model_ids = [506, 86, 79, 113, 583, 751, 407, 166, 47, 610]


def high_entropy_ce():
    model_loader = FmnistLenetPretrainedModelLoader()
    ce_search = HighEntropyThresholdOptimizer(num_of_epochs=100,
                                              accuracy_weight=1.0,
                                              mac_weight=0.0,
                                              model_loader=model_loader,
                                              model_id=424,
                                              val_ratio=0.5,
                                              image_output_path=output_path,
                                              num_of_gmm_components_per_block=[2, 2],
                                              entropy_threshold_percentiles=[0.95, 0.95],
                                              entropy_threshold_counts_after_percentiles=[5, 5],
                                              random_seed=10,
                                              are_entropy_thresholds_fixed=True)
    ce_search.run()
    print("X")


def run_sigmoid_gmm_ce():
    model_loader = FmnistLenetPretrainedModelLoader()
    ce_search = SigmoidGmmCeThresholdOptimizer(num_of_epochs=100,
                                               accuracy_weight=1.0,
                                               mac_weight=0.0,
                                               model_loader=model_loader,
                                               model_id=424,
                                               val_ratio=0.5,
                                               image_output_path=output_path,
                                               entropy_threshold_counts=[5, 5],
                                               num_of_gmm_components_per_block=[2, 2],
                                               random_seed=10,
                                               are_entropy_thresholds_fixed=False)
    ce_search.run()
    print("X")


def run_categorical_ce():
    model_loader = FmnistLenetPretrainedModelLoader()
    ce_search = CategoricalCeThresholdOptimizer(num_of_epochs=100,
                                                accuracy_weight=1.0,
                                                mac_weight=1.0,
                                                model_loader=model_loader,
                                                model_id=424,
                                                val_ratio=0.5,
                                                image_output_path=output_path,
                                                entropy_threshold_counts=[5, 5],
                                                entropy_bins_count=50,
                                                probability_bins_count=50, random_seed=10,
                                                are_entropy_thresholds_fixed=False)
    ce_search.run()
    print("X")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    run_sigmoid_gmm_ce()
