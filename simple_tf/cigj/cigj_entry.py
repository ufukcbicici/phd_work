import tensorflow as tf

from data_handling.cifar_dataset import CifarDataSet
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.fashion_net import fashion_net_cigj
from simple_tf.global_params import GlobalConstants


def cigj_training():
    dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0]
    classification_dropout_probs = [0.0]
    decision_dropout_probs = [0.0]
    sess = tf.Session()
    jungle = Jungle(
        node_build_funcs=[fashion_net_cigj.f_root_func,
                          fashion_net_cigj.f_l1_func,
                          fashion_net_cigj.f_l2_func,
                          fashion_net_cigj.f_l3_func,
                          fashion_net_cigj.f_leaf_func],
        h_funcs=[fashion_net_cigj.h_l1_func], grad_func=None, threshold_func=None, residue_func=None, summary_func=None,
        degree_list=[1, 3, 3, 3, 1], dataset=dataset)
    jungle.print_trellis_structure()

cigj_training()
# batch_size = tf.placeholder(dtype=tf.int32)
# dist = tf.distributions.Categorical(probs=[0.3, 0.4, 0.3])
# samples = dist.sample(batch_size)
# one_hot_samples = tf.one_hot(indices=samples, depth=3, axis=-1)
#
# sess = tf.Session()
# res = sess.run([samples, one_hot_samples], feed_dict={batch_size: 100000})
# print("X")
