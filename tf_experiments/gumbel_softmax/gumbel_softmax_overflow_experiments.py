import tensorflow as tf
import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_gumbel_softmax import JungleGumbelSoftmax
from simple_tf.info_gain import InfoGainLoss

arrs = UtilityFuncs.load_npz("C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf//cigj//tensors")
activations = arrs["activations"]
probs = arrs["probs"]
oneHotLabelTensor = arrs["oneHotLabelTensor"]
activation_grads = arrs["activation_grads"]
prob_grads = arrs["prob_grads"]

probs_tf = tf.placeholder(name="probs", dtype=tf.float32, shape=probs.shape)
labels_tf = tf.placeholder(name="labels", dtype=tf.float32, shape=oneHotLabelTensor.shape)
# probs = tf.nn.softmax(logits)
ig = InfoGainLoss.get_loss(p_n_given_x_2d=probs_tf, p_c_given_x_2d=labels_tf, balance_coefficient=1.0)
z_samples = JungleGumbelSoftmax.sample_from_gumbel_softmax(probs=probs_tf,
                                                           temperature=0.01,
                                                           z_sample_count=100,
                                                           batch_size=probs.shape[0],
                                                           child_count=probs.shape[1])
grads = tf.gradients(ig, probs_tf)
sess = tf.Session()

# x = np.array([-38.396988, -19.412645, 51.50944]).reshape((1, 3))
# l = np.zeros(10)
# l[3] = 1.0
# l = l.reshape((1, 10))
results = sess.run([ig, probs_tf, z_samples], feed_dict={probs_tf: probs, labels_tf: oneHotLabelTensor})
print("X")
