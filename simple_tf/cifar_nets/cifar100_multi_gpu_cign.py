from simple_tf.cifar_nets.cifar100_cign import Cifar100_Cign
from simple_tf.cign.cign_multi_gpu import CignMultiGpu


class Cifar100_MultiGpuCign(CignMultiGpu):
    def __init__(self, degree_list, dataset):
        node_build_funcs = [Cifar100_Cign.root_func, Cifar100_Cign.l1_func, Cifar100_Cign.leaf_func]
        super().__init__(node_build_funcs, None, None, None, None, degree_list, dataset)

    def