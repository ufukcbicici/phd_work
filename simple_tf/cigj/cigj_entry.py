import tensorflow as tf
import numpy as np

from data_handling.cifar_dataset import CifarDataSet
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.fashion_net import fashion_net_cigj
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants


def cigj_training():
    dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0]
    classification_dropout_probs = [0.0]
    decision_dropout_probs = [0.0]
    jungle = Jungle(
        node_build_funcs=[FashionNetCigj.f_conv_layer_func,
                          FashionNetCigj.f_conv_layer_func,
                          FashionNetCigj.f_conv_layer_func,
                          FashionNetCigj.f_fc_layer_func,
                          FashionNetCigj.f_leaf_func],
        h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
        grad_func=None,
        threshold_func=FashionNetCigj.threshold_calculator_func,
        residue_func=None, summary_func=None,
        degree_list=GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST, dataset=dataset)
    sess = jungle.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    histogram = np.zeros(shape=(GlobalConstants.EVAL_BATCH_SIZE, 3))
    for i in range(1000):
        results, _ = jungle.eval_network(sess=sess, dataset=dataset, use_masking=True)
        print("X")

    # jungle.print_trellis_structure()


cigj_training()



# with tf.control_dependencies([shape_assign_op]):
#     set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
#     with tf.control_dependencies([set_batch_size_op]):
#         x = tf.identity(shape_tensor)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run([x], feed_dict={sparse_tensor: sparse_arr, batch_size_tensor: batch_size})
# print("X")
# square_sparse = tf.square(sparse_tensor)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# with tf.control_dependencies(shape_assign_op):
#     # with tf.control_dependencies(set_batch_size_op):
#     results = sess.run([square_sparse], feed_dict={sparse_tensor: sparse_arr, indices_tensor: indices,
#                                                    batch_size_tensor: batch_size})
#     print("X")

# zero_index = tf.constant(0)


# tf.scatter_update()
# shape_assign_op = tf.assign(shape_tensor, tf.shape(sparse_tensor))
# set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
# with tf.control_dependencies(shape_assign_op):
# with tf.control_dependencies(set_batch_size_op):
# square = tf.square(sparse_arr)


# batch_size = tf.placeholder(dtype=tf.int32)
# prob_tensor = tf.placeholder(dtype=tf.float32)
# prob_arr = np.array([[0.3, 0.4, 0.3], [0.7, 0.1, 0.2], [0.25, 0.25, 0.5], [0.95, 0.05, 0.05], [0.1, 0.2, 0.7]])
# dist = tf.distributions.Categorical(probs=prob_tensor)
# samples = dist.sample()
# one_hot_samples = tf.one_hot(indices=samples, depth=3, axis=-1)
# sess = tf.Session()
# samples_arr = None
# for i in range(100000):
#     print(i)
#     res = sess.run([samples, one_hot_samples], feed_dict={batch_size: 100000, prob_tensor: prob_arr})
#     if i == 0:
#         samples_arr = res[0]
#         samples_arr = np.expand_dims(samples_arr, axis=1)
#     else:
#         curr_samples = res[0]
#         curr_samples = np.expand_dims(curr_samples, axis=1)
#         samples_arr = np.concatenate((samples_arr, curr_samples), axis=1)
# print("X")
