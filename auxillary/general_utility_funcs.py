import os
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import itertools
from datetime import datetime

from auxillary.parameters import DecayingParameter, FixedParameter
from simple_tf.global_params import GlobalConstants
from tensorflow.python.client import device_lib


class UtilityFuncs:
    def __init__(self):
        pass

    @staticmethod
    def print(string):
        print(string)

    @staticmethod
    def compare_floats(f1, f2, eps=1e-10):
        abs_dif = abs(f1 - f2)
        return abs_dif <= eps

    @staticmethod
    def get_timestamp():
        date_obj = datetime.now()
        timestamp_str = "{0}_{1}_{2}_{3}_{4}_{5}_{6}" \
            .format(date_obj.date().year, date_obj.date().month,
                    date_obj.date().day, date_obj.time().hour,
                    date_obj.time().minute, date_obj.time().second, date_obj.time().microsecond)
        return timestamp_str

    @staticmethod
    def set_tuple_element(target_tuple, target_index, value):
        tuple_as_list = list(target_tuple)
        tuple_as_list[target_index] = value
        new_tuple = tuple(tuple_as_list)
        return new_tuple

    @staticmethod
    def get_absolute_path(script_file, relative_path):
        script_dir = os.path.dirname(script_file)
        absolute_path = script_dir + "/" + relative_path  # os.path.join(script_dir, relative_path)
        absolute_path = absolute_path.replace("\\", "/")
        return absolute_path

    @staticmethod
    def get_cartesian_product(list_of_lists):
        cartesian_product = list(itertools.product(*list_of_lists))
        return cartesian_product

    @staticmethod
    def tf_safe_flatten(input_tensor):
        flattened_dim = np.prod(np.array(input_tensor.get_shape().as_list())[1:])
        flattened = tf.reshape(input_tensor, shape=(-1, flattened_dim))
        return flattened
        # shape_tensor = tf.shape(input_tensor)
        # flat_shape = tf.stack([shape_tensor[0], tf.reduce_prod(shape_tensor[1:])], axis=0)
        # flattened_tensor = tf.reshape(input_tensor, shape=flat_shape)
        # return flattened_tensor

    @staticmethod
    def get_max_val_acc_from_baseline_results(results_folder):
        file_score_tuples = []
        files = [f for f in listdir(results_folder) if isfile(join(results_folder, f))]
        max_score = 0.0
        max_file_name = ""
        for file in files:
            path = "{0}\\{1}".format(results_folder, file)
            with open(path, 'r') as myfile:
                data = myfile.read().replace('\n', '')
                last_equality_pos = data.rfind("=")
                last_index = len(data)
                val = data[last_equality_pos + 1: last_index]
                score = float(val)
                if score > max_score:
                    max_score = score
                    max_file_name = file
                file_score_tuples.append((file, score))
        print("max_score:{0}".format(max_score))
        print("max_file_name:{0}".format(max_file_name))
        file_score_tuples = sorted(file_score_tuples, key=lambda pair: pair[1])
        for tpl in file_score_tuples:
            print("file:{0}".format(tpl[0]))
            print("score:{0}".format(tpl[1]))
        return max_score, max_file_name

    @staticmethod
    def get_entropy_of_set(label_set):
        sample_count = label_set.shape[0]
        label_dict = {}
        for label in label_set:
            if not (label in label_dict):
                label_dict[label] = 0
            label_dict[label] += 1
        entropy = 0.0
        for label, quantity in label_dict.items():
            probability = float(quantity) / float(sample_count)
            entropy -= probability * np.log2(probability)
        return entropy

    @staticmethod
    def create_parameter_from_train_program(parameter_name, train_program):
        value_dict = train_program.load_settings_for_property(property_name=parameter_name)
        if value_dict["type"] == "DecayingParameter":
            param_object = DecayingParameter.from_training_program(name=parameter_name, training_program=train_program)
        elif value_dict["type"] == "FixedParameter":
            param_object = FixedParameter.from_training_program(name=parameter_name, training_program=train_program)
        else:
            raise Exception("Unknown parameter type.")
        return param_object

    @staticmethod
    def load_npz(file_name):
        filename = file_name + ".npz"
        try:
            npzfile = np.load(filename)
        except:
            raise Exception('Tree file {0} not found'.format(filename))
        return npzfile

    @staticmethod
    def save_npz(file_name, arr_dict):
        np.savez(file_name, **arr_dict)

    @staticmethod
    def calculate_mac_of_computation(num_of_input_channels,
                                     height_of_input_map,
                                     width_of_input_map,
                                     height_of_filter,
                                     width_of_filter,
                                     num_of_output_channels,
                                     convolution_stride,
                                     type="conv"):
        if type == "conv":
            C = num_of_input_channels
            H = height_of_input_map
            W = width_of_input_map
            R = height_of_filter
            S = width_of_filter
            M = num_of_output_channels
            # E = height_of_output_map
            # F = width_of_output_map
            U = convolution_stride
            E = (H - R + U) / U
            F = (W - S + U) / U
            cost = M * F * E * R * S * C
            # for m in range(M):
            #     for x in range(F):
            #         for y in range(E):
            #             for i in range(R):
            #                 for j in range(S):
            #                     for k in range(C):
            #                         cost += 1
            return cost
        elif type == "depth_seperable":
            C = num_of_input_channels
            H = height_of_input_map
            W = width_of_input_map
            R = height_of_filter
            S = width_of_filter
            M = num_of_output_channels
            # E = height_of_output_map
            # F = width_of_output_map
            U = convolution_stride
            E = (H - R + U) / U
            F = (W - S + U) / U
            cost = E * F * C * (R * S + M)
            return cost
        elif type == "fc":
            C = num_of_input_channels
            H = 1
            W = 1
            R = 1
            S = 1
            M = num_of_output_channels
            E = 1
            F = 1
        else:
            raise NotImplementedError()
        cost = M * F * E * R * S * C
        return cost

    @staticmethod
    def get_available_devices(only_gpu=True):
        local_device_protos = device_lib.list_local_devices()
        if only_gpu:
            return [x.name for x in local_device_protos if x.device_type == 'GPU']
        return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']

    @staticmethod
    def concat_to_np_array_dict(dct, key, array):
        if key not in dct:
            if not np.isscalar(array):
                dct[key] = array
            else:
                scalar = array
                dct[key] = np.array(scalar)
        else:
            if not np.isscalar(array):
                dct[key] = np.concatenate((dct[key], array))
            else:
                scalar = array
                dct[key] = np.append(dct[key], scalar)

    @staticmethod
    def concat_to_np_array_dict_v2(dct, key, array):
        if key not in dct:
            if not np.isscalar(array):
                dct[key] = [array]
            else:
                scalar = array
                dct[key] = np.array(scalar)
        else:
            if not np.isscalar(array):
                dct[key].append(array)
            else:
                scalar = array
                dct[key] = np.append(dct[key], scalar)

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @staticmethod
    def get_modes_from_distribution(distribution, percentile_threshold):
        cumulative_prob = 0.0
        sorted_distribution = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        modes = set()
        for tpl in sorted_distribution:
            if cumulative_prob < percentile_threshold:
                modes.add(tpl[0])
                cumulative_prob += tpl[1]
        return modes

    @staticmethod
    def get_variable_name(name, node, prefix=""):
        return "{0}Node{1}_{2}".format(prefix, node.index, name)

    @staticmethod
    def convert_labels_to_one_hot(labels, max_label):
        assert (len(labels.shape) == 1)
        one_hot_labels = np.zeros((labels.shape[0], max_label))
        one_hot_labels[np.arange(labels.shape[0]), labels[:]] = 1
        return one_hot_labels

    @staticmethod
    def distribute_evenly_to_threads(num_of_threads, list_to_distribute):
        thread_dict = {}
        curr_thread_id = 0
        for item in list_to_distribute:
            if curr_thread_id not in thread_dict:
                thread_dict[curr_thread_id] = []
            thread_dict[curr_thread_id].append(item)
            curr_thread_id = (curr_thread_id + 1) % num_of_threads
        return thread_dict

    @staticmethod
    def calculate_distribution_entropy(distribution):
        log_prob = np.log(distribution + GlobalConstants.INFO_GAIN_LOG_EPSILON)
        prob_log_prob = distribution * log_prob
        entropy = -1.0 * np.sum(prob_log_prob)
        return entropy

    @staticmethod
    def create_variable(name, shape, dtype, initializer, trainable=True):
        if GlobalConstants.USE_MULTI_GPU:
            with tf.device(GlobalConstants.GLOBAL_PINNING_DEVICE):
                try:
                    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
                except ValueError as e:
                    if str(e) == "If initializer is a constant, do not specify shape.":
                        if GlobalConstants.USE_MULTI_GPU:
                            var = tf.get_variable(name, initializer=initializer, dtype=dtype, trainable=trainable)
                    else:
                        raise e
                except:
                    raise NotImplementedError()
                return var
        else:
            try:
                var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
            except ValueError as e:
                if str(e) == "If initializer is a constant, do not specify shape.":
                    var = tf.get_variable(name, initializer=initializer, dtype=dtype, trainable=trainable)
            except:
                raise NotImplementedError()
            return var

    # Expand the indices to contain multiplier effect.
    @staticmethod
    def expand_index_list(indices, index_multiplier):
        expanded_indices = []
        for idx in indices:
            edx = idx * index_multiplier
            expanded_indices.extend(list(range(edx, edx + index_multiplier, 1)))
        assert len(set(expanded_indices)) == len(expanded_indices)
        return np.array(expanded_indices)

    @staticmethod
    def global_average_pooling(net_input):
        net = net_input
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        net_shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        return net

    @staticmethod
    def vectorize_with_gap(X):
        sess = tf.Session()
        batch_size = 10000
        all_indices = np.arange(X.shape[0])
        X_shape = list(X.shape)
        X_shape[0] = None
        x_input = tf.placeholder(dtype=tf.float32, shape=X_shape, name="x_input")
        net = x_input
        x_output = UtilityFuncs.global_average_pooling(net_input=net)
        X_arr = []
        for batch_idx in range(0, all_indices.shape[0], batch_size):
            X_batch = X[batch_idx: batch_idx + batch_size]
            X_formatted_batch = sess.run([x_output], feed_dict={x_input: X_batch})[0]
            X_arr.append(X_formatted_batch)
        X_formatted = np.concatenate(X_arr, axis=0)
        return X_formatted