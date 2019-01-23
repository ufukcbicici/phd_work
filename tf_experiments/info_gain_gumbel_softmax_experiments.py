import tensorflow as tf
import numpy as np
from auxillary.constants import DatasetTypes
from auxillary.parameters import DiscreteParameter
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


def build_conv_layer(x, filter_size, num_of_input_channels, num_of_output_channels, name_suffix=""):
    # OK
    conv_weights = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                            stddev=0.1, dtype=tf.float32))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[num_of_output_channels], dtype=tf.float32))
    conv = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool


def build_fc_layer(x, output_dim):
    input_dim = x.get_shape().as_list()[-1]
    fc_weights = tf.Variable(tf.truncated_normal(
        [input_dim, output_dim], stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
    fc_biases = tf.Variable(tf.constant(0.1, shape=[output_dim], dtype=GlobalConstants.DATA_TYPE))
    fc = tf.nn.relu(tf.matmul(x, fc_weights) + fc_biases)
    return fc


def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


def entropy(p):
    assert len(p.shape) == 1
    entropy = -np.sum(p * np.log(p))
    return entropy


channel_count = 32
child_count = 3
temperature = 0.1
batch_size = 125
z_sample_count = 100
balance_coeff = 1.5
dataset = FashionMnistDataSet(validation_sample_count=0)
epsilon_probability = 0.001
# Experimental maximum IG
z_mean_best = (1.0 / child_count) * np.ones(shape=(child_count,))
choices = np.random.choice(child_count, dataset.get_label_count())
list_z_mean_x_given_label = []
for choice_z in choices:
    z_mean_x_given_label = epsilon_probability * np.ones(shape=(child_count,))
    z_mean_x_given_label[choice_z] = 1.0 - 2.0 * epsilon_probability
    list_z_mean_x_given_label.append(z_mean_x_given_label)
e_0 = entropy(z_mean_best)
e_1 = np.mean(np.array([entropy(p) for p in list_z_mean_x_given_label]))

dataTensor = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="dataTensor")
labelTensor = tf.placeholder(tf.int32, shape=(None,), name="dataTensor")
indices_tensor = tf.placeholder(name="indices_tensor", dtype=tf.int32)
batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)
z_sample_count_tensor = tf.placeholder(name="z_sample_count", dtype=tf.int32)
temperature_tensor = tf.placeholder(name="temperature_tensor", dtype=tf.float32)
balance_coefficient_tensor = tf.placeholder(name="balance_coefficient", dtype=tf.float32)
transformed_x = build_conv_layer(x=dataTensor, filter_size=5, num_of_input_channels=1,
                                 num_of_output_channels=channel_count)
reduced_x = global_avg_pool(x=transformed_x)
h_layer = build_fc_layer(x=reduced_x, output_dim=child_count)
probs = tf.nn.softmax(h_layer)

uniform = tf.distributions.Uniform(low=0.0, high=1.0)
uniform_sample = uniform.sample(sample_shape=(batch_size_tensor, z_sample_count_tensor, child_count))
gumbel_sample = -1.0 * tf.log(-1.0 * tf.log(uniform_sample))
# Concrete
log_probs = tf.log(probs)
log_probs = tf.expand_dims(log_probs, dim=1)
pre_transform = log_probs + gumbel_sample
temp_divided = pre_transform / temperature_tensor
logits = tf.math.exp(temp_divided)
nominator = tf.expand_dims(tf.reduce_sum(logits, axis=2), dim=2)
z_samples = logits / nominator
# ExpConcrete
log_sum_exp = tf.expand_dims(tf.reduce_logsumexp(temp_divided, axis=2), axis=2)
y_samples = temp_divided - log_sum_exp
z_samples_stable = tf.exp(y_samples)
# Information Gain
z_mean = tf.reduce_mean(z_samples_stable, [0, 1])
z_per_sample = tf.reduce_mean(z_samples_stable, [1])
label_equalities = []
list_z_x_given_y = []
list_of_sums = []
list_z_mean_x_given_y = []
H_z_x = InfoGainLoss.calculate_entropy(prob_distribution=z_mean)[0]
list_H_z_given_y = []
for class_label in range(dataset.get_label_count()):
    equality = tf.expand_dims(tf.expand_dims(tf.cast(tf.equal(class_label, labelTensor), tf.float32), axis=1), axis=2)
    sum_label = tf.reduce_sum(equality)
    z_x_given_y = tf.multiply(equality, z_samples_stable)
    label_equalities.append(equality)
    list_of_sums.append(sum_label)
    list_z_x_given_y.append(z_x_given_y)
    z_sum_x_given_y = tf.reduce_sum(z_x_given_y, [0, 1])
    z_mean_x_given_y = (1.0 / (tf.cast(z_sample_count_tensor, tf.float32) * sum_label)) * z_sum_x_given_y
    list_z_mean_x_given_y.append(z_mean_x_given_y)
    p_y = sum_label / tf.cast(batch_size, tf.float32)
    list_H_z_given_y.append(p_y * InfoGainLoss.calculate_entropy(prob_distribution=z_mean_x_given_y)[0])
information_gain_loss = -1.0 * (balance_coefficient_tensor * H_z_x - tf.add_n(list_H_z_given_y))
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

fullbatch = dataset.get_next_batch(batch_size=batch_size)
results = sess.run([transformed_x, reduced_x, z_samples, z_samples_stable, z_mean,
                    labelTensor, label_equalities, z_per_sample, list_of_sums, list_z_x_given_y, list_z_mean_x_given_y,
                    H_z_x, list_H_z_given_y, information_gain_loss],
                   feed_dict={dataTensor: fullbatch.samples,
                              labelTensor: fullbatch.labels,
                              batch_size_tensor: batch_size,
                              z_sample_count_tensor: z_sample_count,
                              temperature_tensor: temperature,
                              balance_coefficient_tensor: balance_coeff})
globalCounter = tf.Variable(0, trainable=False)
INITIAL_LR = 0.1
LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                             value=INITIAL_LR,
                                             schedule=[(40000, 0.01),
                                                       (60000, 0.001),
                                                       (80000, 0.0001)])
boundaries = [tpl[0] for tpl in LEARNING_RATE_CALCULATOR.schedule]
values = [INITIAL_LR]
values.extend([tpl[1] for tpl in LEARNING_RATE_CALCULATOR.schedule])
learningRate = tf.train.piecewise_constant(globalCounter, boundaries, values)
optimizer = tf.train.MomentumOptimizer(learningRate, 0.9).minimize(information_gain_loss, global_step=globalCounter)
init = tf.global_variables_initializer()
sess.run(init)
min_ig = 0.0
for epoch in range(250):
    dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
    while True:
        minibatch = dataset.get_next_batch(batch_size=batch_size)
        _, lr, cntr = sess.run([optimizer, learningRate, globalCounter], feed_dict={dataTensor: minibatch.samples,
                                                                                    labelTensor: minibatch.labels,
                                                                                    batch_size_tensor: batch_size,
                                                                                    z_sample_count_tensor: z_sample_count,
                                                                                    temperature_tensor: temperature,
                                                                                    balance_coefficient_tensor: balance_coeff})
        results = sess.run([transformed_x, reduced_x, z_samples, z_samples_stable, z_mean, probs,
                            labelTensor, label_equalities, z_per_sample, list_of_sums, list_z_x_given_y,
                            list_z_mean_x_given_y,
                            H_z_x, list_H_z_given_y, information_gain_loss],
                           feed_dict={dataTensor: minibatch.samples,
                                      labelTensor: minibatch.labels,
                                      batch_size_tensor: batch_size,
                                      z_sample_count_tensor: z_sample_count,
                                      temperature_tensor: temperature,
                                      balance_coefficient_tensor: balance_coeff})
        ig_loss = results[-1]
        if np.isnan(ig_loss):
            print("NAN!!!!")
        if ig_loss < min_ig:
            min_ig = ig_loss
        print("iteration={0} lr={1} ig_loss={2} min_ig_loss:{3} z_mean:{4}".format(cntr, lr, ig_loss, min_ig,
                                                                                   results[4]))

print("X")
