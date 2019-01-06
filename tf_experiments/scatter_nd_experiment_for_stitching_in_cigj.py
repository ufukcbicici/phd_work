import tensorflow as tf
import numpy as np

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
results = []
for exp_index in range(1000):
    batch_size = max(1, int(1000.0 * np.random.uniform()))
    sparse_length = max(1, int(batch_size * np.random.uniform()))
    sparse_arr = np.random.uniform(low=-1.0, high=1.0, size=(sparse_length, 14, 14, 32))
    indices = np.array(sorted(np.random.choice(a=batch_size, size=sparse_length, replace=False).tolist()))
    indices_tensor = tf.placeholder(name="indices", dtype=tf.int32)
    sparse_tensor = tf.placeholder(name="sparse_arr", dtype=tf.float32)
    batch_size_tensor = tf.placeholder(name="batch_size", dtype=tf.int32)
    shape_tensor = tf.Variable(name="shape", trainable=False, initial_value=[0] * 4)
    shape_assign_op = tf.assign(shape_tensor, tf.shape(sparse_tensor))


    def func(set_shape_op, indices, updates, shape):
        with tf.control_dependencies([set_shape_op]):
            set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
            with tf.control_dependencies([set_batch_size_op]):
                indices = tf.expand_dims(indices, -1)
                scatter = tf.scatter_nd(indices, updates, shape)
                return scatter, shape
    stitch_op, shape = func(set_shape_op=shape_assign_op, indices=indices_tensor, updates=sparse_tensor,
                            shape=shape_assign_op)
    res = sess.run([stitch_op, shape], feed_dict={sparse_tensor: sparse_arr, indices_tensor: indices,
                                                  batch_size_tensor: batch_size})
    indices_list = indices.tolist()
    indices_set_c = sorted(list(set(range(batch_size)).difference(set(indices_list))))
    assert len(indices_list) + len(indices_set_c) == batch_size
    l1 = [np.allclose(res[0][indices_list[i]], sparse_arr[i]) for i in range(len(indices_list))]
    l2 = [np.array_equal(res[0][indices_set_c[i]], np.zeros(shape=res[0][indices_set_c[i]].shape)) for i in range(len(indices_set_c))]
    assert all(l1) and all(l2)
    results.append(all(l1) and all(l2))
    print("Experiment:{0} BatchSize:{1} NonZeroSize:{2} Result:{3}".format(exp_index, batch_size, sparse_length,
                                                                           (all(l1) and all(l2))))

# l1 = [res[0][indices]]

# print(res[0].shape)
# print(res[1])
final_result = all(results)
print("Final Result:{0}".format(final_result))