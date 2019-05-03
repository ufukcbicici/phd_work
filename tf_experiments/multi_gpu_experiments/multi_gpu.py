import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.contrib.nccl.ops import gen_nccl_ops

from auxillary.general_utility_funcs import UtilityFuncs

available_devices = UtilityFuncs.get_available_devices()
batch_size = 250
input_tensors = []
for device_id, device in enumerate(available_devices):
    input_tensor = tf.placeholder(name="input_{0}".format(device_id), dtype=tf.float32, shape=(batch_size, 32, 32, 20))
    input_tensors.append(input_tensor)

input_tensors_numpy = [np.random.uniform(0, 1.0, (batch_size, 32, 32, 20)) for _ in range(len(available_devices))]
gen_nccl_ops.nccl_all_reduce

print("X")