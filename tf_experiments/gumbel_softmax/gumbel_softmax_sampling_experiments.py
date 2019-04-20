import tensorflow as tf
import numpy as np
from data_handling.fashion_mnist import FashionMnistDataSet
from auxillary.constants import DatasetTypes
from simple_tf.cigj.jungle_gumbel_softmax import JungleGumbelSoftmax

child_count = 3
temperature = 0.01
z_sample_count = 1000
dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
sample_count = dataset.get_current_sample_count()
x = dataset.get_next_batch(batch_size=sample_count)

# Placeholder
x_tensor = tf.placeholder(name="x", dtype=tf.float32,
                          shape=[None, dataset.get_image_size(), dataset.get_image_size(), 1])
batch_size_tensor = tf.placeholder(name="batch_size", dtype=tf.int32)
z_sample_count_tensor = tf.placeholder(name="z_sample_count", dtype=tf.int32)
temperature_tensor = tf.placeholder(name="temperature", dtype=tf.float32)

# Calculate Probability p(F|x)
x_flat = tf.contrib.layers.flatten(x_tensor)
hidden_layer = tf.layers.dense(inputs=x_flat, units=64, activation=tf.nn.relu)
h_layer = tf.layers.dense(inputs=hidden_layer, units=child_count, activation=tf.nn.relu)
probs = tf.nn.softmax(h_layer)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

z_samples = \
    JungleGumbelSoftmax.sample_from_gumbel_softmax(probs=probs, temperature=temperature_tensor,
                                                   z_sample_count=z_sample_count_tensor, batch_size=batch_size_tensor,
                                                   child_count=child_count)

results = sess.run([z_samples, probs], feed_dict={x_tensor: x.samples,
                                                  batch_size_tensor: sample_count,
                                                  z_sample_count_tensor: z_sample_count,
                                                  temperature_tensor: temperature})

print("X")
