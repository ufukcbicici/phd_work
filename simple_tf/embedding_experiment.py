import tensorflow as tf
import numpy as np

from algorithms.softmax_compresser import SoftmaxCompresser
from data_handling.fashion_mnist import FashionMnistDataSet

dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
modes = {0,1,3}
label_mapping = SoftmaxCompresser.get_compressed_probability_mapping(modes=modes, dataset=dataset)


training_labels = dataset.trainingLabels
mapped_labels = np.zeros(shape=training_labels.shape, dtype=training_labels.dtype)

# Manual
for i in range(training_labels.shape[0]):
    mapped_labels[i] = label_mapping[training_labels[i]]

# Tf
sess = tf.Session()
labels_tf = tf.placeholder(tf.int64, shape=(None,))
label_mapping_tf = tf.placeholder(tf.int64)
embedding_op = tf.nn.embedding_lookup(params=label_mapping_tf, ids=labels_tf)

result = sess.run([embedding_op], feed_dict={labels_tf: training_labels, label_mapping_tf: label_mapping})
print("Is result Equal:{0}".format(np.array_equal(result[0], mapped_labels)))
assert np.array_equal(result[0], mapped_labels)
# embedding_op = tf.nn.embedding_lookup(params=)

print("X")


