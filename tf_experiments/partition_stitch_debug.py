import tensorflow as tf

from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.uncategorized.global_params import GlobalConstants


def build_conv_layer(input, filter_size, num_of_input_channels, num_of_output_channels, name_suffix=""):
    # OK
    conv_weights = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE))
    conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool


def h_transform(input, dropout_prob, h_feature_size, pool_size):
    h_net = input
    # Parametric Average Pooling if the input layer is convolutional
    assert len(h_net.get_shape().as_list()) == 2 or len(h_net.get_shape().as_list()) == 4
    if len(h_net.get_shape().as_list()) == 4:
        h_net = tf.nn.avg_pool(h_net, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1],
                               padding='SAME')
        # h_net = UtilityFuncs.tf_safe_flatten(input_tensor=h_net)
        h_net = tf.contrib.layers.flatten(h_net)

    feature_size = h_net.get_shape().as_list()[-1]
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, h_feature_size],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[h_feature_size], dtype=GlobalConstants.DATA_TYPE))
    h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
    h_net = tf.nn.relu(h_net)
    h_net = tf.nn.dropout(h_net, keep_prob=dropout_prob)
    ig_feature = h_net
    return ig_feature


dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)

dataTensor = tf.placeholder(GlobalConstants.DATA_TYPE,
                            shape=(None, dataset.get_image_size(),
                                   dataset.get_image_size(),
                                   dataset.get_num_of_channels()),
                            name="dataTensor")
labelTensor = tf.placeholder(tf.int64, shape=(None,), name="labelTensor")
dropoutProb = tf.placeholder(tf.float32, name="dropoutProb")
isTrain = tf.placeholder(tf.int32, name="isTrain")
softmaxDecay = tf.placeholder(tf.float32, name="softmaxDecay")
batchSize = tf.placeholder(tf.int32, name="batchSize")
batchIndices = tf.range(batchSize)
child_count = 3
outputs = []
# Node 0
f_net = build_conv_layer(input=dataTensor, filter_size=5, num_of_input_channels=1, num_of_output_channels=32)
outputs.append(f_net)
# Node 1 (h)
partitions = []
ig_feature = h_transform(input=f_net, dropout_prob=dropoutProb, h_feature_size=128, pool_size=7)
outputs.append(ig_feature)
ig_feature_size = ig_feature.get_shape().as_list()[-1]
hyperplane_weights = tf.Variable(
    tf.truncated_normal([ig_feature_size, child_count], stddev=0.1, seed=GlobalConstants.SEED,
                        dtype=GlobalConstants.DATA_TYPE))
hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[child_count], dtype=GlobalConstants.DATA_TYPE))
ig_feature = tf.layers.batch_normalization(inputs=ig_feature, momentum=GlobalConstants.BATCH_NORM_DECAY,
                                           training=tf.cast(isTrain, tf.bool))
activations = tf.matmul(ig_feature, hyperplane_weights) + hyperplane_biases
decayed_activation = activations / tf.reshape(softmaxDecay, (1,))
p_F_given_x = tf.nn.softmax(decayed_activation)
indices_tensor = tf.argmax(p_F_given_x, axis=1, output_type=tf.int32)
f_conditionIndices = tf.dynamic_partition(data=batchIndices, partitions=indices_tensor, num_partitions=child_count)
f_net_parts = tf.dynamic_partition(data=f_net, partitions=indices_tensor, num_partitions=child_count)
f_label_parts = tf.dynamic_partition(data=labelTensor, partitions=indices_tensor, num_partitions=child_count)
outputs.append(indices_tensor)
outputs.extend(f_conditionIndices)
outputs.extend(f_net_parts)
outputs.extend(f_label_parts)

# Nodes 2,3,4
f_outputs = []
for sibling_index, node_index in enumerate([2, 3, 4]):
    f_input = f_net_parts[sibling_index]
    f_output = build_conv_layer(input=f_input, filter_size=5, num_of_input_channels=32, num_of_output_channels=24)
    f_outputs.append(f_output)
    # outputs.append(f_output)

# Node 5
f_stitched = tf.dynamic_stitch(indices=f_conditionIndices, data=f_outputs)
f_labels_stitched = tf.dynamic_stitch(indices=f_conditionIndices, data=f_label_parts)
outputs.append(f_stitched)
outputs.append(f_labels_stitched)

# Run
sess = tf.Session()
minibatch = dataset.get_next_batch(batch_size=GlobalConstants.EVAL_BATCH_SIZE)
feed_dict = {dataTensor: minibatch.samples, labelTensor: minibatch.labels,
             dropoutProb: 1.0, isTrain: 0, softmaxDecay: 50.0, batchSize: GlobalConstants.EVAL_BATCH_SIZE}
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10000):
    results = sess.run(outputs, feed_dict=feed_dict)
    print("{0} runned.".format(i))