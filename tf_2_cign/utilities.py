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