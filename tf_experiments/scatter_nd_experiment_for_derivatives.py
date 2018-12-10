import tensorflow as tf
import numpy as np
import time

from auxillary.general_utility_funcs import UtilityFuncs

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
batch_size = 125
child_count = 4


def stitch_tensor(index, input_tensor, indices, batch_size, name):
    shape_tensor_name = "{0}_shape_tensor_{1}".format(name, index)
    shape_tensor = tf.Variable(name=shape_tensor_name, trainable=False,
                               initial_value=[0] * len(input_tensor.get_shape().as_list()))
    shape_assign_op = tf.assign(shape_tensor, tf.shape(input_tensor))
    with tf.control_dependencies([shape_assign_op]):
        # Set the first element as the batch size
        set_batch_size_op = tf.assign(shape_tensor[0], batch_size)
        with tf.control_dependencies([set_batch_size_op]):
            # Get indices of the f_node in the appropriate format
            indices = tf.expand_dims(indices, -1)
            # Obtain the sparse output
            sparse_output = tf.scatter_nd(indices, input_tensor, shape_tensor)
            return sparse_output


for exp_index in range(1000):
    t0 = time.time()
    activation_arr = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, child_count))
    x = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 14, 14, 32))
    x_squared = np.square(x)
    activation_tensor = tf.placeholder(name="activation_tensor", dtype=tf.float32, shape=activation_arr.shape)
    input_tensor = tf.placeholder(name="input_tensor", dtype=tf.float32, shape=x.shape)
    batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)
    index_tensor = tf.range(batch_size)
    arg_max_tensor = tf.argmax(activation_tensor, axis=1)
    one_hot_samples = tf.one_hot(indices=arg_max_tensor, depth=child_count, axis=-1, dtype=tf.int64)
    t1 = time.time()
    # Mask
    sparse_x_list = []
    sparse_indices_list = []
    sparse_transformed_x_list = []
    for mask_index in range(child_count):
        mask_tensor = one_hot_samples[:, mask_index]
        masked_x = tf.boolean_mask(input_tensor, mask_tensor)
        transformed_masked_x = tf.square(masked_x)
        masked_indices = tf.boolean_mask(index_tensor, mask_tensor)
        sparse_x = stitch_tensor(index=mask_index, input_tensor=masked_x, indices=masked_indices,
                                 batch_size=batch_size_tensor,
                                 name="masked_x")
        sparse_indices = stitch_tensor(index=mask_index, input_tensor=masked_indices, indices=masked_indices,
                                       batch_size=batch_size_tensor,
                                       name="masked_indices")
        sparse_transformed_x = stitch_tensor(index=mask_index, input_tensor=transformed_masked_x,
                                             indices=masked_indices,
                                             batch_size=batch_size_tensor,
                                             name="transformed_masked_x")
        sparse_x_list.append(sparse_x)
        sparse_indices_list.append(sparse_indices)
        sparse_transformed_x_list.append(sparse_transformed_x)
    original_x = tf.add_n(sparse_x_list)
    indices_summed = tf.add_n(sparse_indices_list)
    transformed_x = tf.add_n(sparse_transformed_x_list)
    loss = tf.reduce_sum(transformed_x)
    t2 = time.time()
    grads = tf.gradients(loss, input_tensor)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    t3 = time.time()
    results = sess.run([original_x, indices_summed, transformed_x, grads, loss],
                       feed_dict={activation_tensor: activation_arr, input_tensor: x, batch_size_tensor: batch_size})
    t4 = time.time()
    res1 = np.allclose(x, results[0])
    res2 = np.array_equal(np.arange(batch_size), results[1])
    res3 = np.allclose(x_squared, results[2])
    res4 = np.allclose(2.0 * x, results[3])
    t5 = time.time()
    assert res1 and res2 and res3 and res4
    print("{0} - res1:{1} res2:{2} res3:{3} res4:{4}".format(exp_index, res1, res2, res3, res4))
    print("t1-t0:{0} t2-t1:{1} t3-t2:{2} t4-t3:{3} t5-t4:{4}".format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4))
    tf.reset_default_graph()


























    #
    #     sparse_length = max(1, int(batch_size * np.random.uniform()))
    #     sparse_arr = np.random.uniform(low=-1.0, high=1.0, size=(sparse_length, 14, 14, 32))
    #     indices = np.array(sorted(np.random.choice(a=batch_size, size=sparse_length, replace=False).tolist()))
    #     indices_tensor = tf.placeholder(name="indices", dtype=tf.int32)
    #     sparse_tensor = tf.placeholder(name="sparse_arr", dtype=tf.float32)
    #     batch_size_tensor = tf.placeholder(name="batch_size", dtype=tf.int32)
    #     shape_tensor = tf.Variable(name="shape", trainable=False, initial_value=[0] * 4)
    #     shape_assign_op = tf.assign(shape_tensor, tf.shape(sparse_tensor))
    #
    #
    #     def func(set_shape_op, indices, updates, shape):
    #         with tf.control_dependencies([set_shape_op]):
    #             set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
    #             with tf.control_dependencies([set_batch_size_op]):
    #                 indices = tf.expand_dims(indices, -1)
    #                 scatter = tf.scatter_nd(indices, updates, shape)
    #                 return scatter, shape
    #     stitch_op, shape = func(set_shape_op=shape_assign_op, indices=indices_tensor, updates=sparse_tensor,
    #                             shape=shape_assign_op)
    #     res = sess.run([stitch_op, shape], feed_dict={sparse_tensor: sparse_arr, indices_tensor: indices,
    #                                                   batch_size_tensor: batch_size})
    #     indices_list = indices.tolist()
    #     indices_set_c = sorted(list(set(range(batch_size)).difference(set(indices_list))))
    #     assert len(indices_list) + len(indices_set_c) == batch_size
    #     l1 = [np.allclose(res[0][indices_list[i]], sparse_arr[i]) for i in range(len(indices_list))]
    #     l2 = [np.array_equal(res[0][indices_set_c[i]], np.zeros(shape=res[0][indices_set_c[i]].shape)) for i in range(len(indices_set_c))]
    #     assert all(l1) and all(l2)
    #     results.append(all(l1) and all(l2))
    #     print("Experiment:{0} BatchSize:{1} NonZeroSize:{2} Result:{3}".format(exp_index, batch_size, sparse_length,
    #                                                                            (all(l1) and all(l2))))
    #
    # # l1 = [res[0][indices]]
    #
    # # print(res[0].shape)
    # # print(res[1])
    # final_result = all(results)
    # print("Final Result:{0}".format(final_result))
