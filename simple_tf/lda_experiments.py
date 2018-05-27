import tensorflow as tf
import matplotlib.pyplot as plt

from algorithms.lda_loss import LdaLoss
from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.toy_dataset import ToyDataset


def visualize_data(title, dataset, x, l, id, s_between, s_inner):
    plt.figure(id)
    plt.title("{0} \n s_between:{1} \n s_inner:{2}".format(title, s_between, s_inner))
    for c in range(dataset.get_label_count()):
        mask = l == c
        plt.plot(x[mask, 0], x[mask, 1], '.', label="class {0}".format(c))
    plt.legend()
    plt.savefig("{0}".format(title))
    plt.close()
    # plt.show()


toy_dataset = ToyDataset(validation_sample_count=0)

samples = toy_dataset.trainingSamples
labels = toy_dataset.trainingLabels

tf_data = tf.placeholder(dtype=tf.float32, shape=UtilityFuncs.set_tuple_element(samples.shape, 0, None))
tf_labels = tf.placeholder(dtype=tf.int64, shape=[None])

# lda_loss = LdaLoss.get_loss(data=tf_data, labels=tf_labels, dataset=toy_dataset)
#
# print("X")
# feed_dict = {tf_data: samples, tf_labels: labels}
# sess = tf.Session()
# results = sess.run([lda_loss], feed_dict)

layer_count = 3
layer_width = 1024
batch_size = 312
epoch_count = 100000

net = tf_data
for l_index in range(layer_count):
    net = tf.layers.dense(net, layer_width, activation=tf.nn.relu)
lda_layer = tf.layers.dense(net, samples.shape[1], activation=None)
lda_loss, total_between_class_variance, total_inner_class_variance, \
    between_class_covariance_matrix, inner_class_covariance_matrix, inner_var_list, inner_class_cov_matrices = \
    LdaLoss.get_loss(data=lda_layer, labels=tf_labels, dataset=toy_dataset)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.000001, global_step,
                                           2500,
                                           0.2,
                                           staircase=True)
optimizer = tf.train.MomentumOptimizer(0.0000001, 0.9).minimize(lda_loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
toy_dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
visualize_data(title="Start", dataset=toy_dataset, x=toy_dataset.trainingSamples, l=toy_dataset.trainingLabels, id=0,
               s_between=0, s_inner=0)
iteration = 0
for epoch_id in range(epoch_count):
    toy_dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
    print("*************Epoch {0}*************".format(epoch_id))
    while True:
        x_, l_, indices_list, one_hot_labels = toy_dataset.get_next_batch(batch_size=batch_size)
        run_ops = [optimizer, lda_loss]
        feed_dict = {tf_data: x_, tf_labels: l_}
        results = sess.run(run_ops, feed_dict=feed_dict)
        iteration += 1
        print("Lda Loss:{0}".format(results[1]))
        if toy_dataset.isNewEpoch:
            # toy_dataset.visualize_dataset(dataset_type=)
            toy_dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            feed_dict = {tf_data: toy_dataset.trainingSamples, tf_labels: toy_dataset.trainingLabels}
            results = sess.run([lda_layer, total_between_class_variance, total_inner_class_variance,
                                between_class_covariance_matrix, inner_class_covariance_matrix, inner_var_list,
                                inner_class_cov_matrices],
                               feed_dict=feed_dict)
            lda_x = results[0]
            s_between = results[1]
            s_inner = results[2]
            SB = results[3]
            SW = results[4]
            if iteration % 10 == 0:
                visualize_data(title="After_Epoch_{0}".format(epoch_id), dataset=toy_dataset, x=lda_x,
                               l=toy_dataset.trainingLabels, id=epoch_id+1,
                               s_between=s_between, s_inner=s_inner)
            break
print("X")
