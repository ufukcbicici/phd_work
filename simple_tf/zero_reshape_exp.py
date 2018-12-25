import tensorflow as tf
import numpy as np

from simple_tf.global_params import GlobalConstants

data_tensor = tf.placeholder(GlobalConstants.DATA_TYPE, shape=(None, 28, 28, 100), name="dataTensor")
indices_tensor = tf.ones(100, dtype=tf.int32)
# indices_tensor = tf.concat([tf.zeros(dtype=tf.int32, shape=(60,)), tf.ones(dtype=tf.int32, shape=(40,))],
#                            axis=0)
parts = tf.dynamic_partition(data=data_tensor, partitions=indices_tensor, num_partitions=2)
shape_0 = tf.shape(parts[0])
shape_1 = tf.shape(parts[1])
flattened_0 = tf.contrib.layers.flatten(parts[0])
flattened_1 = tf.contrib.layers.flatten(parts[1])
new_shape_0 = tf.stack([shape_0[0], tf.reduce_prod(shape_0[1:])], axis=0)
new_shape_1 = tf.stack([shape_1[0], tf.reduce_prod(shape_1[1:])], axis=0)

reshaped_0 = tf.reshape(parts[0], shape=new_shape_0)
reshaped_1 = tf.reshape(parts[1], shape=new_shape_1)
sum = tf.reduce_sum(reshaped_0) + tf.reduce_sum(reshaped_1)
grads = tf.gradients(sum, [data_tensor])

arr = np.random.uniform(size=(100, 28, 28, 100))
sess = tf.Session()
# results = sess.run([flattened_0, flattened_1], feed_dict={data_tensor: arr})
results = sess.run([shape_0, shape_1, new_shape_0, new_shape_1, reshaped_0, reshaped_1, grads],
                   feed_dict={data_tensor: arr})
print("X")
