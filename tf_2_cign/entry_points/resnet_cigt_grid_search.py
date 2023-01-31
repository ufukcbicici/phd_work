import tensorflow as tf
import numpy as np
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.cigt.cigt import Cigt
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.cigt.resnet_cigt import ResnetCigt
from tf_2_cign.data.cifar10 import Cifar10
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.resnet_cigt_constants import ResnetCigtConstants
from tf_2_cign.utilities.utilities import Utilities

# Hyper-parameters
from tf_2_cign.fashion_net.fashion_cign_rl import FashionCignRl
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    Cifar10.TF_RNG = tf.random.Generator.from_seed(123, alg='philox')
    DbLogger.log_db_path = DbLogger.tetam_cigt_2
    # 5e-4,
    # 0.0005
    weight_decay = 10 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    weight_decay = sorted(weight_decay)
    param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])

    for param_tpl in param_grid:
        ResnetCigtConstants.classification_wd = param_tpl[0]
        cifar10 = Cifar10(batch_size=ResnetCigtConstants.batch_size, validation_size=0)

        with tf.device("GPU"):
            run_id = DbLogger.get_run_id()
            resnet_cigt = ResnetCigt(run_id=run_id, model_definition="Resnet-110 Cigt Thin Baseline Grid Search")

            explanation = resnet_cigt.get_explanation_string()
            DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

            resnet_cigt.fit(x=cifar10.trainDataset,
                            validation_data=cifar10.testDataset,
                            epochs=ResnetCigtConstants.epoch_count)





    # ResnetCigt.create_default_config_json()
    # print("X")
