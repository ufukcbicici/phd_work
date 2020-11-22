from collections import deque

import numpy as np
import tensorflow as tf


class RpropOptimizer:

    @staticmethod
    def train(grads_list):
        alpha = 1.2
        beta = 0.5
        lr = 0.001
        min_lr = 1e-10
        loss_history = deque(maxlen=10000)
        grad_history = deque(maxlen=10000)
