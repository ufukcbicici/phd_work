import tensorflow as tf
import numpy as np

from simple_tf.cigj.jungle import Jungle

probs = tf.placeholder(name="probs", dtype=tf.float32)
batch_size = tf.placeholder(name="batch_size", dtype=tf.int32)
category_count = tf.placeholder(name="category_count", dtype=tf.int32)

batch_size_ = 250
category_count_ = 3

unnormalized_probs_ = np.random.uniform(low=0.0, high=100.0, size=(batch_size_, category_count_))
probs_ = unnormalized_probs_ / np.reshape(np.sum(unnormalized_probs_, axis=1),
                                          newshape=(unnormalized_probs_.shape[0], 1))

histogram = np.zeros(shape=(batch_size_, category_count_))

selected_indices = Jungle.sample_from_categorical(probs=probs, batch_size=batch_size, category_count=category_count)

sess = tf.Session()
for i in range(1000000):
    if i % 100 == 0:
        print(i)
    results = sess.run([selected_indices], feed_dict={probs: probs_, batch_size: batch_size_,
                                                      category_count: category_count_})
    histogram[np.arange(histogram.shape[0]), results[0]] += 1
sampled_probs = histogram / np.reshape(np.sum(histogram, axis=1), newshape=(histogram.shape[0], 1))
print("X")