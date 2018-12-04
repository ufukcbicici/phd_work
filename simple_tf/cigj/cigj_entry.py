import tensorflow as tf

from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.global_params import GlobalConstants


def cigj_training():
    sess = tf.Session()
    dataset = CifarDataSet(session=sess, validation_sample_count=0, load_validation_from=None)
    jungle = Jungle(node_build_funcs=[None]*5, grad_func=None, threshold_func=None, residue_func=None, summary_func=None,
                    degree_list=[1, 3, 3, 3, 1], dataset=dataset)
    jungle.print_trellis_structure()


cigj_training()
