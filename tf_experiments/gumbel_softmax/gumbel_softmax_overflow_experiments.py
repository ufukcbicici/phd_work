import tensorflow as tf
import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_gumbel_softmax import JungleGumbelSoftmax
from simple_tf.info_gain import InfoGainLoss

# arrs = UtilityFuncs.load_npz("C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf//cigj//tensors")
arrs = UtilityFuncs.load_npz("C://Users//ufuk.bicici//Desktop//PHD//phd_work//tensors")

activations = arrs["activations"]
probs = arrs["probs"]
oneHotLabelTensor = arrs["oneHotLabelTensor"]
activation_grads = arrs["activation_grads"]
prob_grads = arrs["prob_grads"]

faulty_prob = np.reshape(np.array([0.0000000e+00, 1.0000000e+00, 3.7425826e-20]), newshape=(1, 3))

probs_tf = tf.placeholder(name="probs", dtype=tf.float32, shape=(None, probs.shape[1]))
labels_tf = tf.placeholder(name="labels", dtype=tf.float32, shape=(None, oneHotLabelTensor.shape[1]))
# probs = tf.nn.softmax(logits)
ig = InfoGainLoss.get_loss(p_n_given_x_2d=probs_tf, p_c_given_x_2d=labels_tf, balance_coefficient=1.0)
z_samples = JungleGumbelSoftmax.sample_from_gumbel_softmax(probs=probs_tf,
                                                           temperature=0.01,
                                                           z_sample_count=100,
                                                           batch_size=faulty_prob.shape[0],
                                                           child_count=faulty_prob.shape[1])
sample_sum = tf.reduce_sum(z_samples)
grads_ig = tf.gradients(ig, probs_tf)
grads_samples = tf.gradients(sample_sum, probs_tf)
sess = tf.Session()

# x = np.array([-38.396988, -19.412645, 51.50944]).reshape((1, 3))
# l = np.zeros(10)
# l[3] = 1.0
# l = l.reshape((1, 10))
results = sess.run([probs_tf, z_samples, grads_samples], feed_dict={probs_tf: faulty_prob,
                                                                    labels_tf: oneHotLabelTensor})
print("X")
