from tf_2_cign.cign import Cign
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign

# Hyper-parameters
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

input_dims = (28, 28, 3)
degree_list = [2, 2]
batch_size = 128
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
node_build_funcs = [FashionCign.inner_func, FashionCign.inner_func, FashionCign.leaf_func]

if __name__ == "__main__":
    fashion_mnist = FashionMnist(batch_size=batch_size)
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=softmax_decay_initial,
                                                      decay_coefficient=softmax_decay_coefficient,
                                                      decay_period=softmax_decay_period,
                                                      decay_min_limit=softmax_decay_min_limit)
    cign = FashionCign(input_dims=input_dims,
                       node_degrees=degree_list,
                       filter_counts=filter_counts,
                       kernel_sizes=kernel_sizes,
                       hidden_layers=hidden_layers,
                       decision_drop_probability=decision_drop_probability,
                       classification_drop_probability=drop_probability,
                       decision_wd=decision_wd,
                       classification_wd=classification_wd,
                       decision_dimensions=decision_dimensions,
                       node_build_funcs=node_build_funcs,
                       class_count=10,
                       information_gain_balance_coeff=1.0,
                       softmax_decay_controller=softmax_decay_controller)
    cign.build_network()

    cign.train(dataset=fashion_mnist, epoch_count=epoch_count)




