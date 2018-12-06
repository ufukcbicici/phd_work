import tensorflow as tf
import numpy as np

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
        h_funcs=[fashion_net_cigj.h_l1_func],
        grad_func=None,
        threshold_func=fashion_net_cigj.threshold_calculator_func,
        residue_func=None, summary_func=None,
        degree_list=[1, 3, 3, 3, 1], dataset=dataset)
    init = tf.global_variables_initializer()
    sess.run(init)
    jungle.eval_network(sess=sess, dataset=dataset, use_masking=True)

    # jungle.print_trellis_structure()


# cigj_training()

batch_size = 250
sparse_length = 110
sparse_arr = np.random.uniform(low=-1.0, high=1.0, size=(sparse_length, 14, 14, 32))
indices = np.array(sorted(np.random.choice(a=batch_size, size=sparse_length, replace=False).tolist()))

sparse_tensor = tf.placeholder(name="sparse_arr", dtype=tf.float32)
indices_tensor = tf.placeholder(name="indices", dtype=tf.int32)
batch_size_tensor = tf.placeholder(name="batch_size", dtype=tf.int32)
shape_tensor = tf.Variable(name="shape", trainable=False, initial_value=[0] * 4)
shape_assign_op = tf.assign(shape_tensor, tf.shape(sparse_tensor))
# set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


def func(dependencies, indices, updates, shape):
    with tf.control_dependencies(dependencies):
        scatter = tf.scatter_nd(indices, updates, shape)
        return scatter


stitch_op = func(dependencies=[shape_assign_op], indices=indices_tensor, updates=sparse_tensor, shape=shape_assign_op)
# square_op2 = tf.square(square_op1)
res = sess.run([shape_tensor], feed_dict={sparse_tensor: sparse_arr, indices_tensor: indices,
                                          batch_size_tensor: batch_size})
print("X")

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
