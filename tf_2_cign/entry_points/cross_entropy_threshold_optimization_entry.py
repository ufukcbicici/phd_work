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
from tf_2_cign.cigt.cross_entropy_optimizers.cross_entropy_threshold_optimizer import CrossEntropySearchOptimizer
from tf_2_cign.cigt.cross_entropy_optimizers.sigmoid_gmm_ce_threshold_optimizer import SigmoidGmmCeThresholdOptimizer
from tf_2_cign.cigt.model_loaders.fmnist_lenet_pretrained_model_loader import FmnistLenetPretrainedModelLoader

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # requests.post("http://65.21.32.250:5000/api/get_images", data=json.dumps(
    #     [3, 5, 15]))

    DbLogger.log_db_path = DbLogger.home_asus
    output_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..", "tf_2_cign",
                               "cigt", "image_outputs")

    model_loader = FmnistLenetPretrainedModelLoader()
    ce_search = SigmoidGmmCeThresholdOptimizer(num_of_epochs=100,
                                               accuracy_mac_balance_coeff=1.0,
                                               model_loader=model_loader,
                                               model_id=424,
                                               val_ratio=0.25,
                                               image_output_path=output_path,
                                               entropy_threshold_counts=[5, 5],
                                               num_of_gmm_components_per_block=[2, 2],
                                               random_seed=10)
    ce_search.run()
    print("X")
