import json

import tensorflow as tf
import numpy as np
import os
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.cigt.cigt import Cigt
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign
from tf_2_cign.softmax_decay_algorithms.hyperbolic_decay_algorithm import HyperbolicDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities

# Hyper-parameters
from tf_2_cign.fashion_net.fashion_cign_rl import FashionCignRl
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm


class BayesianOptimizer:
    def __init__(self, xi, init_points, n_iter, val_ratio):
        self.optimization_bounds_continuous = None
        self.xi = xi
        self.init_points = init_points
        self.n_iter = n_iter
        self.val_ratio = val_ratio

    def get_old_and_new_log_file_names(self, log_file_root_path, log_file_name):
        relevant_files = []
        for file_name in os.listdir(log_file_root_path):
            if file_name.startswith(log_file_name) and file_name.endswith(".json"):
                log_order = int(file_name[file_name.rindex("_num") + 4: -5])
                relevant_files.append((file_name, log_order))
        if len(relevant_files) > 0:
            old_log_num = max([tpl[1] for tpl in relevant_files])
            new_log_num = old_log_num + 1
            old_log_file = log_file_name + "_num" + str(old_log_num) + ".json"
            old_log_path = os.path.join(log_file_root_path, old_log_file)
        else:
            old_log_path = None
            new_log_num = 0
        new_log_file = log_file_name + "_num" + str(new_log_num) + ".json"
        new_log_path = os.path.join(log_file_root_path, new_log_file)
        return old_log_path, new_log_path

    def fit(self, log_file_root_path, log_file_name):
        old_log_path, new_log_path = self.get_old_and_new_log_file_names(log_file_root_path, log_file_name)

        optimizer = BayesianOptimization(
            f=self.cost_function,
            pbounds=self.optimization_bounds_continuous,
            verbose=10
        )

        logger = JSONLogger(path=new_log_path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        if old_log_path is not None:
            load_logs(optimizer, logs=[old_log_path])
            # Get the count of total past experiments
            f = open(old_log_path)
            exp_count = len(f.readlines())
            init_points_after_log_read = max(0, self.init_points - exp_count)
            n_iter_after_log_read = self.n_iter - max(0, exp_count - self.init_points)
            f.close()
        else:
            init_points_after_log_read = self.init_points
            n_iter_after_log_read = self.n_iter
        if n_iter_after_log_read < 0:
            return

        optimizer.maximize(
            n_iter=n_iter_after_log_read,
            init_points=init_points_after_log_read,
            acq="ei",
            xi=0.01)

    def cost_function(self, **kwargs):
        pass
