import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.contrib.nccl.ops import gen_nccl_ops

from auxillary.general_utility_funcs import UtilityFuncs

def experiment():
    available_devices = UtilityFuncs.get_available_devices()
    print(available_devices)
    batch_size = 250
    gpu_count = 1
    input_tensors = []
    means = []
    sync_means = []
    with tf.device("/cpu:0"):
        with tf.variable_scope("Shared"):
            for device_id in range(gpu_count):
                with tf.device("/gpu:{0}".format(device_id)):
                    # Input Tensors
                    input_tensor = tf.placeholder(name="input_{0}".format(device_id),
                                                  dtype=tf.float32, shape=(batch_size, 32, 32, 20))
                    input_tensors.append(input_tensor)
                    # Mean (per GPU)
                    mean = tf.reduce_mean(input_tensor, axis=[0, 1, 2])
                    means.append(mean)
                    # Mean (Over all GPUs)
                    sync_mean = gen_nccl_ops.nccl_all_reduce(
                              input=mean,
                              reduction='sum',
                              num_devices=gpu_count,
                              shared_name="Shared") * (1.0 / gpu_count)
                    sync_means.append(sync_mean)

    input_tensors_numpy = \
        [np.random.uniform(0, 1.0, (batch_size, 32, 32, 20)) for _ in range(gpu_count)]
    sess = tf.Session()
    res = sess.run([means, sync_means],
                   feed_dict={input_tensors[device_id]: input_arr
                              for device_id, input_arr in enumerate(input_tensors_numpy)})
    print("X")


    # input_tensors_numpy = [np.random.uniform(0, 1.0, (batch_size, 32, 32, 20)) for _ in range(len(available_devices))]
    # # gen_nccl_ops.nccl_all_reduce

    print("X")