from data_handling.mnist_data_set import MnistDataSet
import tensorflow as tf


k = 3
D = MnistDataSet.MNIST_SIZE * MnistDataSet.MNIST_SIZE
threshold = 0.3
feature_count = 32
epsilon = 0.000001
batch_size = 100

dataset = MnistDataSet(validation_sample_count=5000)
dataset.load_dataset()

samples, labels, indices_list = dataset.get_next_batch()
initializer = tf.contrib.layers.xavier_initializer()
x = tf.placeholder(tf.float32, name="x")
flat_x = tf.reshape(x, [-1, D])
W1 = tf.get_variable(name="W1", shape=[D, 500], initializer=initializer, dtype=tf.float32)
reg_W1 = tf.nn.l2_loss(W1)
b1 = tf.get_variable(name="b1", shape=[500], initializer=initializer, dtype=tf.float32)
reg_b1 = tf.nn.l2_loss(b1)
W2 = tf.get_variable(name="W2", shape=[500, 10], initializer=initializer, dtype=tf.float32)
reg_W2 = tf.nn.l2_loss(W2)
b2 = tf.get_variable(name="b2", shape=[10], initializer=initializer, dtype=tf.float32)
reg_b2 = tf.nn.l2_loss(b2)


a1 = tf.matmul(flat_x, W1) + b1
h1 = tf.nn.tanh(a1)

a2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.tanh(a2)

last_layer = tf.reduce_mean(h2)
loss_list = [last_layer, reg_W1, reg_b1, reg_W2, reg_b2]
loss = tf.add_n(loss_list)
grads = tf.gradients(loss, [W1, b1, W2, b2])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

results = sess.run([loss ,grads], feed_dict={x: samples})
sum_grad = tf.gradients(last_layer, [W1])
l2_grad = tf.gradients(reg_W1, [W1])
results2 = sess.run([sum_grad, l2_grad], feed_dict={x: samples})

print("X")
