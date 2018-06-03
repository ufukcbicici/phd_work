import os
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import itertools

from auxillary.parameters import DecayingParameter, FixedParameter
from simple_tf.global_params import GlobalConstants


class UtilityFuncs:
    def __init__(self):
        pass

    @staticmethod
    def compare_floats(f1, f2, eps=1e-10):
        abs_dif = abs(f1 - f2)
        return abs_dif <= eps

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
    def concat_to_np_array_dict(dct, key, array):
        if key not in dct:
            dct[key] = array
        else:
            dct[key] = np.concatenate((dct[key], array))

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
    def create_mlp_layers(input, layer_dims, use_dropout_list, dropout_prob_tensor, variable_name_prefix,
                          weight_bias_list=None):
        curr_layer = input
        variable_list = []
        if weight_bias_list is None:
            assert len(layer_dims) == len(use_dropout_list)
            layer_count = len(layer_dims)
            for layer_index in range(layer_count):
                layer_dim = layer_dims[layer_index]
                use_dropout = use_dropout_list[layer_index]
                if use_dropout:
                    curr_layer = tf.nn.dropout(curr_layer, keep_prob=dropout_prob_tensor)
                input_dim = curr_layer.get_shape().as_list()[-1]
                weights = tf.Variable(
                    tf.truncated_normal([input_dim, layer_dim], stddev=0.1,
                                        seed=GlobalConstants.SEED,
                                        dtype=GlobalConstants.DATA_TYPE),
                    name="{0}_weights_{1}".format(variable_name_prefix, layer_index))
                bias = tf.Variable(tf.constant(0.1, shape=[layer_dim],
                                               dtype=GlobalConstants.DATA_TYPE),
                                   name="{0}_biases_{1}".format(variable_name_prefix, layer_index))
                variable_list.extend([weights, bias])
                curr_layer = tf.nn.relu(tf.matmul(curr_layer, weights) + bias)
            output = curr_layer
            return output, variable_list
        else:
            assert len(weight_bias_list) == len(use_dropout_list)
            layer_count = len(weight_bias_list)
            for layer_index in range(layer_count):
                weights = weight_bias_list[layer_index]["weights"]
                bias = weight_bias_list[layer_index]["bias"]
                use_dropout = use_dropout_list[layer_index]
                if use_dropout:
                    curr_layer = tf.nn.dropout(curr_layer, keep_prob=dropout_prob_tensor)
                variable_list.extend([weights, bias])
                curr_layer = tf.nn.relu(tf.matmul(curr_layer, weights) + bias)
            output = curr_layer
            return output, variable_list
