import tensorflow as tf
import numpy as np

from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.uncategorized.global_params import GlobalConstants
from data_handling.data_set import DataSet


dataTensor = tf.placeholder(GlobalConstants.DATA_TYPE,
                                 shape=(GlobalConstants.BATCH_SIZE, GlobalConstants.IMAGE_SIZE,
                                        GlobalConstants.IMAGE_SIZE,
                                        GlobalConstants.NUM_CHANNELS))
labelTensor = tf.placeholder(tf.int64, shape=(GlobalConstants.BATCH_SIZE,))

conv1_weights = tf.Variable(
    tf.truncated_normal([GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_1_SIZE,
                         GlobalConstants.NUM_CHANNELS, GlobalConstants.FASHION_NUM_FILTERS_1], stddev=0.1,
                        seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE), name="conv1_weight")
conv1_biases = tf.Variable(
    tf.constant(0.1, shape=[GlobalConstants.FASHION_NUM_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
    name="conv1_bias")

conv2_weights = tf.Variable(
    tf.truncated_normal([GlobalConstants.FASHION_FILTERS_2_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
                         GlobalConstants.FASHION_NUM_FILTERS_1, GlobalConstants.FASHION_NUM_FILTERS_2],
                        stddev=0.1,
                        seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
    name="conv2_weight")
conv2_biases = tf.Variable(
    tf.constant(0.1, shape=[GlobalConstants.FASHION_NUM_FILTERS_2], dtype=GlobalConstants.DATA_TYPE), name="conv2_bias")
conv3_weights = tf.Variable(
    tf.truncated_normal([GlobalConstants.FASHION_FILTERS_3_SIZE, GlobalConstants.FASHION_FILTERS_3_SIZE,
                         GlobalConstants.FASHION_NUM_FILTERS_2, GlobalConstants.FASHION_NUM_FILTERS_3],
                        stddev=0.1,
                        seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE), name="conv3_weight")
conv3_biases = tf.Variable(
    tf.constant(0.1, shape=[GlobalConstants.FASHION_NUM_FILTERS_3], dtype=GlobalConstants.DATA_TYPE), name="conv3_bias")
conv1 = tf.nn.conv2d(dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Second Conv Layer
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Third Conv Layer
conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
flattened = tf.contrib.layers.flatten(pool3)
flat_dimension_size = flattened.get_shape().as_list()[-1]

fc_weights_1 = tf.Variable(tf.truncated_normal(
    [flat_dimension_size, GlobalConstants.FASHION_FC_1],
    stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
    name="fc_weights_1")
fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_FC_1], dtype=GlobalConstants.DATA_TYPE),
                          name="fc_biases_1")
# FC 2 Weights
fc_weights_2 = tf.Variable(tf.truncated_normal(
    [GlobalConstants.FASHION_FC_1, GlobalConstants.FASHION_FC_2],
    stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE), name="fc_weights_2")
fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_FC_2], dtype=GlobalConstants.DATA_TYPE),
                          name="fc_biases_2")
# Softmax Weights
fc_softmax_weights = tf.Variable(
    tf.truncated_normal([GlobalConstants.FASHION_FC_2, GlobalConstants.NUM_LABELS],
                        stddev=0.1,
                        seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE), name="fc_softmax_weights")
fc_softmax_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
                                            dtype=GlobalConstants.DATA_TYPE), name="fc_softmax_biases")
# Fully Connected Layers
hidden_layer_1 = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
dropped_layer_1 = tf.nn.dropout(hidden_layer_1, 0.7)
hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
final_feature = tf.nn.dropout(hidden_layer_2, 0.7)
logits = tf.matmul(final_feature, fc_softmax_weights) + fc_softmax_biases
# cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labelTensor,
#                                                                            logits=logits)
# loss = tf.reduce_mean(cross_entropy_loss_tensor)
# loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)




# results = sess.run(list_of_eval_dicts, feed_dict)
# return results

dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
minibatch = dataset.get_next_batch(batch_size=GlobalConstants.EVAL_BATCH_SIZE)
minibatch = DataSet.MiniBatch(np.expand_dims(minibatch.samples, axis=3), minibatch.labels,
                              minibatch.indices, minibatch.one_hot_labels, minibatch.hash_codes)
feed_dict = {dataTensor: minibatch.samples, labelTensor: minibatch.labels}
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

g = tf.get_default_graph()
run_metadata = tf.RunMetadata()
results = sess.run([logits], feed_dict,
                   options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                   run_metadata=run_metadata)
opts = tf.profiler.ProfileOptionBuilder.float_operation()
flops = tf.profiler.profile(g, run_meta=run_metadata, cmd='op', options=opts)

