import numpy as np
import tensorflow as tf
import pickle


class Utilities:
    def __init__(self):
        pass

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
    def get_variable_name(name, node, prefix=""):
        return "{0}Node{1}_{2}".format(prefix, node.index, name)

    @staticmethod
    def compare_model_list_outputs(l_1, l_2):
        assert len(l_1) == len(l_2)
        for arr_1, arr_2 in zip(l_1, l_1):
            assert isinstance(arr_1, tf.Tensor)
            assert isinstance(arr_2, tf.Tensor)
            is_equal = np.array_equal(arr_1.numpy(), arr_2.numpy())
            if not is_equal:
                return False
        return True

    @staticmethod
    def compare_model_dictionary_outputs(d_1, d_2):
        for k_1 in d_1.keys():
            assert k_1 in d_2
            v_1 = d_1[k_1]
            v_2 = d_2[k_1]
            if isinstance(v_1, dict):
                assert isinstance(v_2, dict)
                is_equal = Utilities.compare_model_dictionary_outputs(d_1=v_1, d_2=v_2)
                if not is_equal:
                    return False
            elif isinstance(v_1, list):
                assert isinstance(v_2, list)
                is_equal = Utilities.compare_model_list_outputs(l_1=v_1, l_2=v_2)
                if not is_equal:
                    return False
            elif isinstance(v_1, tf.Tensor):
                assert isinstance(v_2, tf.Tensor)
                is_equal = np.array_equal(v_1.numpy(), v_2.numpy())
                if not is_equal:
                    return False
            else:
                raise NotImplementedError()
        return True

    @staticmethod
    def compare_model_outputs(output_1, output_2):
        assert isinstance(output_1, dict)
        assert isinstance(output_2, dict)
        assert len(output_1) == len(output_2)

        for _key in output_1.keys():
            assert _key in output_2
            v_1 = output_1[_key]
            v_2 = output_2[_key]
            assert (isinstance(v_1, dict) and isinstance(v_2, dict)) or \
                   (isinstance(v_1, list) and isinstance(v_2, list))
            if isinstance(v_1, dict):
                is_equal = Utilities.compare_model_dictionary_outputs(d_1=v_1, d_2=v_2)
                if not is_equal:
                    return False
            elif isinstance(v_2, list):
                is_equal = Utilities.compare_model_list_outputs(l_1=v_1, l_2=v_2)
                if not is_equal:
                    return False
            else:
                raise NotImplementedError()
        return True

    @staticmethod
    def pickle_save_to_file(path, file_content):
        f = open(path, "wb")
        pickle.dump(file_content, f)
        f.close()

    @staticmethod
    def pickle_load_from_file(path):
        f = open(path, "rb")
        content = pickle.load(f)
        f.close()
        return content

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
    def merge_dict_of_ndarrays(dict_target, dict_to_append):
        # key_set1 = set(dict_target.keys())
        # key_set2 = set(dict_to_append.keys())
        # assert key_set1 == key_set2
        for k in dict_to_append.keys():
            Utilities.concat_to_np_array_dict(dct=dict_target, array=dict_to_append[k], key=k)
