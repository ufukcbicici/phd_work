import numpy as np
import tensorflow as tf

sample_count = 152
logitTensor = tf.placeholder(dtype=tf.float32, shape=(sample_count, 10))
labelTensor = tf.placeholder(dtype=tf.int64, shape=(sample_count,))
scales = tf.Variable(tf.random_normal([1], stddev=0.35))
posterior_probs = tf.nn.softmax(logits=logitTensor)
argmax_label_prediction = tf.argmax(posterior_probs, 1)
comparison_with_labels = tf.equal(x=argmax_label_prediction, y=labelTensor)
comparison_cast = tf.cast(comparison_with_labels, tf.float32)
correct_count = tf.reduce_sum(input_tensor=comparison_cast)
total_count = tf.size(input=comparison_cast)
scaled_probs = posterior_probs * scales
scales_new_value = tf.placeholder(dtype=tf.float32, shape=(1,))
scales_assign = tf.assign(scales, scales_new_value)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
logits = np.random.uniform(0.0, 10.0, size=(sample_count, 10))
label_probs = [1.0 / 10.0] * 10
labels = np.random.multinomial(1, label_probs, (sample_count,)).argmax(axis=1)
for i in range(100):
    results = sess.run([scaled_probs, correct_count, total_count, posterior_probs, scales],
                       feed_dict={logitTensor: logits, labelTensor: labels})
    results2 = sess.run(fetches=scales_assign, feed_dict={scales_new_value: np.array([i])})
    print("X")