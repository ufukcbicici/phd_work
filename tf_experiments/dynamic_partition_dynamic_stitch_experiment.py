import tensorflow as tf
import numpy as np
import time

from auxillary.general_utility_funcs import UtilityFuncs

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
batch_size = 125
child_count = 4

for exp_index in range(1000):
    t0 = time.time()
    activation_arr = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, child_count))
    x = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 14, 14, 32))
    x_squared = np.square(x)
    t1 = time.time()
    activation_tensor = tf.placeholder(name="activation_tensor", dtype=tf.float32, shape=activation_arr.shape)
    input_tensor = tf.placeholder(name="input_tensor", dtype=tf.float32, shape=x.shape)
    batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)
    arg_max_tensor = tf.cast(tf.argmax(activation_tensor, axis=1), tf.int32)
    condition_indices = tf.dynamic_partition(data=tf.range(batch_size_tensor), partitions=arg_max_tensor,
                                             num_partitions=child_count)
    partition_list = tf.dynamic_partition(data=input_tensor, partitions=arg_max_tensor, num_partitions=child_count)
    t2 = time.time()
    transformed_partition_list = []
    for partition in partition_list:
        transformed_partition_list.append(tf.square(partition))
    original_stitched = tf.dynamic_stitch(indices=condition_indices, data=partition_list)
    transformed_stitched = tf.dynamic_stitch(indices=condition_indices, data=transformed_partition_list)
    indices_stitched = tf.dynamic_stitch(indices=condition_indices, data=condition_indices)
    loss = tf.reduce_sum(transformed_stitched)
    grads = tf.gradients(loss, input_tensor)
    t3 = time.time()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    t4 = time.time()
    results = sess.run([original_stitched, transformed_stitched, indices_stitched, grads, condition_indices,
                        partition_list],
                       feed_dict={input_tensor: x, activation_tensor: activation_arr, batch_size_tensor: batch_size})
    t5 = time.time()
    res0 = np.allclose(x, results[0])
    res1 = np.allclose(np.square(x), results[1])
    res2 = np.array_equal(np.arange(batch_size), results[2])
    res3 = np.allclose(2.0 * x, results[3])
    t6 = time.time()
    assert (res0 and res1 and res2 and res3)
    tf.reset_default_graph()
    print("Is Correct:{0}".format((res0 and res1 and res2 and res3)))
    print("t1-t0:{0} t2-t1:{1} t3-t2:{2} t4-t3:{3} t5-t4:{4} t6-t5:{5}"
          .format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))
