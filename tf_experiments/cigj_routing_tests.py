import tensorflow as tf
from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants


class CigjTesting:
    @staticmethod
    def test():
        # Create the dataset and the model
        dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
        node_build_funcs = [FashionNetCigj.f_conv_layer_func,
                            FashionNetCigj.f_conv_layer_func,
                            FashionNetCigj.f_conv_layer_func,
                            FashionNetCigj.f_fc_layer_func,
                            FashionNetCigj.f_leaf_func]
        jungle = Jungle(
            node_build_funcs=node_build_funcs,
            h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
            grad_func=None,
            threshold_func=FashionNetCigj.threshold_calculator_func,
            residue_func=None, summary_func=None,
            degree_list=GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST, dataset=dataset)
        jungle_parameters = set(tf.trainable_variables())
        # Create all root-to-leaf model combinations
        list_of_indices = []
        for degree in GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST:
            list_of_indices.append([i for i in range(degree)])
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=list_of_indices)
        # Build a single root-to-leaf model
        single_path = Jungle(
            node_build_funcs=node_build_funcs,
            h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
            grad_func=None,
            threshold_func=FashionNetCigj.threshold_calculator_func,
            residue_func=None, summary_func=None,
            degree_list=[1] * len(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST), dataset=dataset)
        single_path_variables = set([var for var in tf.trainable_variables() if var not in jungle_parameters])
        shape_set = set(tuple(v.get_shape().as_list()) for v in single_path_variables)
        assert len(shape_set) == len(single_path_variables)
        print("X")


CigjTesting.test()
