import tensorflow as tf

logitTensor = tf.placeholder(dtype=tf.float32, shape=(100, 10))
labelTensor = tf.placeholder(dtype=tf.int64, shape=(100,))
posterior_probs = tf.nn.softmax(logits=logitTensor)
argmax_label_prediction = tf.argmax(posterior_probs, 1)
comparison_with_labels = tf.equal(x=argmax_label_prediction, y=labelTensor)
comparison_cast = tf.cast(comparison_with_labels, tf.float32)
output = tf.reduce_mean(input_tensor=comparison_cast)