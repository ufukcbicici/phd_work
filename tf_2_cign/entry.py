import tensorflow as tf

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign

# Hyper-parameters
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

input_dims = (28, 28, 1)
degree_list = [2, 2]
batch_size = 125
epoch_count = 100
decision_drop_probability = 0.0
drop_probability = 0.0
classification_wd = 0.0
decision_wd = 0.0
softmax_decay_initial = 25.0
softmax_decay_coefficient = 0.9999
softmax_decay_period = 2
softmax_decay_min_limit = 1.0
softmax_decay_controllers = {}

# FashionNet parameters
filter_counts = [32, 32, 32]
kernel_sizes = [5, 5, 1]
hidden_layers = [128, 64]
decision_dimensions = [128, 128]
# node_build_funcs = [FashionCign.inner_func, FashionCign.inner_func, FashionCign.leaf_func]
initial_lr = 0.01
learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                             value=initial_lr,
                                             schedule=[(15000, 0.005),
                                                       (30000, 0.0025),
                                                       (40000, 0.00025)])

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    fashion_mnist = FashionMnist(batch_size=batch_size)
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=softmax_decay_initial,
                                                      decay_coefficient=softmax_decay_coefficient,
                                                      decay_period=softmax_decay_period,
                                                      decay_min_limit=softmax_decay_min_limit)
    with tf.device("GPU"):
        cign = FashionCign(batch_size=batch_size,
                           input_dims=input_dims,
                           node_degrees=degree_list,
                           filter_counts=filter_counts,
                           kernel_sizes=kernel_sizes,
                           hidden_layers=hidden_layers,
                           decision_drop_probability=decision_drop_probability,
                           classification_drop_probability=drop_probability,
                           decision_wd=decision_wd,
                           classification_wd=classification_wd,
                           decision_dimensions=decision_dimensions,
                           class_count=10,
                           information_gain_balance_coeff=1.0,
                           softmax_decay_controller=softmax_decay_controller,
                           learning_rate_schedule=learning_rate_calculator,
                           decision_loss_coeff=1.0)
        # experiment_id = DbLogger.get_run_id()
        # explanation = cign.get_explanation_string()
        # series_id = 0

        cign.build_network()
        cign.train(dataset=fashion_mnist, epoch_count=epoch_count)
