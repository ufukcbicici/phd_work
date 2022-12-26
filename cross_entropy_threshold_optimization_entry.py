import os.path
from multiprocessing import Process

import numpy as np
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
from tf_2_cign.cigt.q_learning_based_post_processing.q_learning_based_post_processing import QLearningRoutingOptimizer

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DbLogger.log_db_path = DbLogger.home_asus
output_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..", "tf_2_cign",
                           "cigt", "image_outputs")

model_ids = [506, 86, 79, 113, 583, 751, 407, 166, 47, 610]
entropy_threshold_counts = [1, 3, 5, 7, 9, 11]
gmm_mode_counts = [1]


def run_on_model(model_id, apply_temperature_to_routing):
    run_count = 50
    model_loader = FmnistLenetPretrainedModelLoader()
    seeds = np.random.randint(low=0, high=1000000, size=(run_count,))
    for seed in seeds:
        for entropy_threshold_count in entropy_threshold_counts:
            for gmm_mode_count in gmm_mode_counts:
                with tf.device('/cpu:0'):
                    ce_search = SigmoidGmmCeThresholdOptimizer(
                        run_id=DbLogger.get_run_id(),
                        num_of_epochs=300,
                        accuracy_weight=1.0,
                        mac_weight=0.0,
                        model_loader=model_loader,
                        model_id=model_id,
                        val_ratio=0.5,
                        image_output_path=output_path,
                        entropy_threshold_counts=[entropy_threshold_count, entropy_threshold_count],
                        num_of_gmm_components_per_block=[gmm_mode_count, gmm_mode_count],
                        random_seed=seed,
                        are_entropy_thresholds_fixed=False,
                        n_jobs=8,
                        apply_temperature_optimization_to_routing_probabilities=apply_temperature_to_routing,
                        apply_temperature_optimization_to_entropies=True)
                    ce_search.run()


def run_on_model_with_seeds(run_id, model_id, seeds):
    model_loader = FmnistLenetPretrainedModelLoader()
    print("Run Id:{0} Seeds:{1}".format(run_id, seeds))
    for seed in seeds:
        for entropy_threshold_count in entropy_threshold_counts:
            for gmm_mode_count in gmm_mode_counts:
                with tf.device('/cpu:0'):
                    ce_search = SigmoidGmmCeThresholdOptimizer(
                        run_id=run_id,
                        num_of_epochs=300,
                        accuracy_weight=1.0,
                        mac_weight=0.0,
                        model_loader=model_loader,
                        model_id=model_id,
                        val_ratio=0.5,
                        image_output_path=output_path,
                        entropy_threshold_counts=[entropy_threshold_count, entropy_threshold_count],
                        num_of_gmm_components_per_block=[gmm_mode_count, gmm_mode_count],
                        random_seed=seed,
                        are_entropy_thresholds_fixed=False,
                        n_jobs=1,
                        apply_temperature_optimization_to_routing_probabilities=False,
                        apply_temperature_optimization_to_entropies=True)
                    ce_search.run()


def experimental(job_id, arr):
    arr.append((job_id, 1, 2, 3))
    arr.append((job_id, 4, 5, 6))
    arr.append((job_id, 7, 8, 9))


def run_on_model_seed_wise_parallelism(model_id):
    n_jobs = 8
    list_of_processes = []
    # run_id_min = DbLogger.get_run_id()
    # print("run_id_min:{0}".format(run_id_min))
    job_array = []
    for job_id in range(n_jobs):
        job_array.append([])

    for job_id in range(n_jobs):
        process = Process(target=experimental, args=(job_id, job_array[job_id]))
        list_of_processes.append(process)
        process.start()

    for process in list_of_processes:
        process.join()

    print("X")

    # for job_id in range(n_jobs):
    #     run_id = run_id_min + job_id
    #     seeds = np.random.randint(low=0, high=1000000, size=(15,))
    #     process = Process(target=run_on_model_with_seeds, args=(run_id, model_id, seeds))
    #     list_of_processes.append(process)
    #     process.start()
    #
    # for process in list_of_processes:
    #     process.join()


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
    ce_search = SigmoidGmmCeThresholdOptimizer(num_of_epochs=300,
                                               accuracy_weight=1.0,
                                               mac_weight=0.0,
                                               model_loader=model_loader,
                                               model_id=610,
                                               val_ratio=0.25,
                                               image_output_path=output_path,
                                               entropy_threshold_counts=[2, 2],
                                               num_of_gmm_components_per_block=[2, 2],
                                               random_seed=966,
                                               are_entropy_thresholds_fixed=False,
                                               n_jobs=8,
                                               apply_temperature_optimization_to_routing_probabilities=False,
                                               apply_temperature_optimization_to_entropies=True)
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


def run_q_net_based_post_processing(model_id):
    model_loader = FmnistLenetPretrainedModelLoader()
    # run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader,
    # model_id, val_ratio, random_seed
    run_id = DbLogger.get_run_id()
    q_learning_routing_optimizer = QLearningRoutingOptimizer(accuracy_weight=1.0,
                                                             mac_weight=0.0,
                                                             run_id=run_id,
                                                             model_id=model_id,
                                                             num_of_epochs=100,
                                                             model_loader=model_loader,
                                                             random_seed=5000,
                                                             val_ratio=0.5,
                                                             max_test_val_diff=0.0020)
    q_learning_routing_optimizer.prepare_q_tables()
    q_learning_routing_optimizer.calibrate_test_and_val_sets()
    q_learning_routing_optimizer.train(epoch_count=100,
                                       batch_size=5000,
                                       input_dimension=128,
                                       lstm_layer_dimensions=[128],
                                       dropout_ratio=0.0)


if __name__ == "__main__":
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # run_on_model(model_id=47, apply_temperature_to_routing=False)
    # run_on_model_seed_wise_parallelism(model_id=47)
    run_q_net_based_post_processing(model_id=47)
