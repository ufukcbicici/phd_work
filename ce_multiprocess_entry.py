import os.path
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import json
import requests
from mpire import WorkerPool

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
from tf_2_cign.utilities.utilities import Utilities

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DbLogger.log_db_path = DbLogger.tetam_tuna_cigt_3
output_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..", "tf_2_cign",
                           "cigt", "image_outputs")

model_ids = [506, 86, 79, 113, 583, 751, 407, 166, 47, 610]


# entropy_threshold_counts = [1, 3, 5, 7, 9, 11]
# gmm_mode_counts = [1]


# def run_on_model(model_id, apply_temperature_to_routing):
#     run_count = 50
#     model_loader = FmnistLenetPretrainedModelLoader()
#     seeds = np.random.randint(low=0, high=1000000, size=(run_count,))
#     for seed in seeds:
#         for entropy_threshold_count in entropy_threshold_counts:
#             for gmm_mode_count in gmm_mode_counts:
#                 with tf.device('/cpu:0'):
#                     ce_search = SigmoidGmmCeThresholdOptimizer(
#                         run_id=DbLogger.get_run_id(),
#                         num_of_epochs=300,
#                         accuracy_weight=1.0,
#                         mac_weight=0.0,
#                         model_loader=model_loader,
#                         model_id=model_id,
#                         val_ratio=0.5,
#                         image_output_path=output_path,
#                         entropy_threshold_counts=[entropy_threshold_count, entropy_threshold_count],
#                         num_of_gmm_components_per_block=[gmm_mode_count, gmm_mode_count],
#                         random_seed=seed,
#                         are_entropy_thresholds_fixed=False,
#                         n_jobs=8,
#                         apply_temperature_optimization_to_routing_probabilities=apply_temperature_to_routing,
#                         apply_temperature_optimization_to_entropies=True)
#                     ce_search.run()
#
#
# def run_on_model_with_seeds(run_id, model_id, seeds):
#     model_loader = FmnistLenetPretrainedModelLoader()
#     print("Run Id:{0} Seeds:{1}".format(run_id, seeds))
#     for seed in seeds:
#         for entropy_threshold_count in entropy_threshold_counts:
#             for gmm_mode_count in gmm_mode_counts:
#                 with tf.device('/cpu:0'):
#                     ce_search = SigmoidGmmCeThresholdOptimizer(
#                         run_id=run_id,
#                         num_of_epochs=300,
#                         accuracy_weight=1.0,
#                         mac_weight=0.0,
#                         model_loader=model_loader,
#                         model_id=model_id,
#                         val_ratio=0.5,
#                         image_output_path=output_path,
#                         entropy_threshold_counts=[entropy_threshold_count, entropy_threshold_count],
#                         num_of_gmm_components_per_block=[gmm_mode_count, gmm_mode_count],
#                         random_seed=seed,
#                         are_entropy_thresholds_fixed=False,
#                         n_jobs=1,
#                         apply_temperature_optimization_to_routing_probabilities=False,
#                         apply_temperature_optimization_to_entropies=True)
#                     ce_search.run()
#
#
# def experimental(job_id, arr):
#     arr.append((job_id, 1, 2, 3))
#     arr.append((job_id, 4, 5, 6))
#     arr.append((job_id, 7, 8, 9))


def run_search(run_id, model_id, random_seed, entropy_threshold_count, gmm_mode_count):
    model_loader = FmnistLenetPretrainedModelLoader()
    # ce_logs_table_rows = [(run_id, model_id, random_seed, entropy_threshold_count, gmm_mode_count)] * 5
    # run_parameters_rows = [(run_id, model_id, random_seed)] * 5
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
            random_seed=random_seed,
            are_entropy_thresholds_fixed=False,
            n_jobs=1,
            apply_temperature_optimization_to_routing_probabilities=False,
            apply_temperature_optimization_to_entropies=True)
        ce_logs_table_rows, run_parameters_rows = ce_search.run()
    return ce_logs_table_rows, run_parameters_rows, ce_search.kvRows, ce_search.explanationString


def run_on_model_seed_wise_parallelism(model_id):
    n_jobs = 8
    seed_count_per_process = 15
    list_of_processes = []
    jsons_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "ce_process_outputs")
    if not os.path.isdir(jsons_path):
        os.mkdir(jsons_path)
    seeds = np.random.randint(low=0, high=100000, size=(n_jobs * seed_count_per_process,)).tolist()
    entropy_threshold_counts = [1, 3, 5, 7, 9]
    model_ids = [model_id]
    gmm_mode_counts = [1, 2, 3]
    run_id = DbLogger.get_run_id()
    cartesian_product = Utilities.get_cartesian_product(list_of_lists=[model_ids,
                                                                       seeds,
                                                                       entropy_threshold_counts,
                                                                       gmm_mode_counts])
    ids_list = run_id + np.arange(len(cartesian_product))
    training_tuples = [(r_id, *tpl) for r_id, tpl in zip(ids_list, cartesian_product)]

    for s_idx in range(0, len(training_tuples), n_jobs):
        params_chunk = training_tuples[s_idx:s_idx + n_jobs]
        assert len(params_chunk) % n_jobs == 0
        with WorkerPool(n_jobs=n_jobs) as pool:
            results = pool.map(run_search, params_chunk, progress_bar=True)
        # Write results into DB.
        # ce_logs_table_rows, run_parameters_rows, ce_search.kvRows, ce_search.explanationString
        for n_job_id in range(n_jobs):
            params = params_chunk[n_job_id]
            job_run_id = params[0]
            # ce_logs_table_rows
            ce_logs_table_rows = results[n_job_id][0]
            DbLogger.write_into_table(rows=ce_logs_table_rows, table="ce_logs_table")
            # run_parameters_rows
            run_parameters_rows = results[n_job_id][1]
            DbLogger.write_into_table(rows=run_parameters_rows, table="run_parameters")
            # ce_search_kv_rows (from explanation string)
            ce_search_kv_rows = results[n_job_id][2]
            DbLogger.write_into_table(rows=ce_search_kv_rows, table="run_parameters")
            # ce_search.explanationString
            ce_search_explanation_string = results[n_job_id][3]
            DbLogger.write_into_table(rows=[(job_run_id, ce_search_explanation_string)],
                                      table="run_meta_data")


if __name__ == "__main__":
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # run_on_model(model_id=47, apply_temperature_to_routing=False)
    run_on_model_seed_wise_parallelism(model_id=47)
