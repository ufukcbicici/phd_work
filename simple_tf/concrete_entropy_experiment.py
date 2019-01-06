import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import warnings
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


def calculate_entropy(prob_distribution):
    log_prob = np.log(prob_distribution + GlobalConstants.INFO_GAIN_LOG_EPSILON)
    # is_inf = tf.is_inf(log_prob)
    # zero_tensor = tf.zeros_like(log_prob)
    # log_prob = tf.where(is_inf, x=zero_tensor, y=log_prob)
    prob_log_prob = prob_distribution * log_prob
    entropy = -1.0 * np.sum(prob_log_prob)
    return entropy


def gumbel_softmax_density(z, probs, temperature):
    n = z.shape[0]
    a = np.math.factorial(n)
    b = np.power(temperature, n-1)
    z_pow_minus_lambda = np.power(z, -temperature)
    z_pow_minus_lambda_minus_one = np.power(z, -temperature-1.0)
    numerator_vec = np.multiply(probs, z_pow_minus_lambda_minus_one)
    denominator_vec = np.multiply(probs, z_pow_minus_lambda)
    denominator_sum = np.sum(denominator_vec)
    numerator_vec = numerator_vec / denominator_sum
    c = np.prod(numerator_vec)
    density = a*b*c
    return density

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    child_count = 3
    dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
    sample_count = dataset.get_current_sample_count()
    x = dataset.get_next_batch(batch_size=sample_count)

    # Regular Entropy
    x_tensor = tf.placeholder(name="x", dtype=tf.float32,
                              shape=[None, dataset.get_image_size(), dataset.get_image_size(), 1])
    x_flat = tf.contrib.layers.flatten(x_tensor)
    hidden_layer = tf.layers.dense(inputs=x_flat, units=64, activation=tf.nn.relu)
    h_layer = tf.layers.dense(inputs=hidden_layer, units=child_count, activation=tf.nn.relu)
    probs = tf.nn.softmax(h_layer)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    results = sess.run([probs], feed_dict={x_tensor: x.samples})
    p_z_given_x = results[0]
    p_z = np.mean(results[0], axis=0)
    log_p_z = np.log(p_z)
    entropy_p_z = calculate_entropy(prob_distribution=p_z)
    # Entropy calculation; the hard way
    p_z_given_x_log_p_z = np.multiply(p_z_given_x, log_p_z)
    entropy_p_z_v2 = -1.0 * np.mean(np.sum(p_z_given_x_log_p_z, axis=1))


    print("p_z={0}".format(p_z))
    print("entropy_p_z={0}".format(entropy_p_z))

    # Gumbel Softmax Entropy
    temperature = 0.01
    z_sample_count = 100
    uniform = tf.distributions.Uniform(low=0.0, high=1.0)
    batch_size_tensor = tf.placeholder(name="batch_size", dtype=tf.int32)
    z_sample_count_tensor = tf.placeholder(name="z_sample_count", dtype=tf.int32)
    temperature_tensor = tf.placeholder(name="temperature", dtype=tf.float32)
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

    results = sess.run([gumbel_sample, uniform_sample, probs, log_probs, pre_transform,
                        temp_divided, logits, nominator, z_samples, y_samples, z_samples_stable],
                       feed_dict={x_tensor: x.samples,
                                  batch_size_tensor: sample_count,
                                  z_sample_count_tensor: z_sample_count,
                                  temperature_tensor: temperature})
    p_z = np.mean(results[2], axis=0)
    print("p_z={0}".format(p_z))
    print("X")
