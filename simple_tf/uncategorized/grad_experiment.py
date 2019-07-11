import tensorflow as tf
import numpy as np

flag = tf.Variable(name="flag", initial_value=1, dtype=tf.int64)
a = tf.Variable(name="a", initial_value=3.0)
b = tf.Variable(name="b", initial_value=5.0)
d = tf.Variable(name="d", initial_value=21.0)
x = tf.where(flag > 0, tf.identity(b), tf.identity(d))
c = a * x
grad = tf.gradients(c, [a, b, d])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

result = sess.run([grad], feed_dict={flag: 0})

print("X")