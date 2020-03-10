from auxillary.general_utility_funcs import UtilityFuncs


def main():
    cost_conv = UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=3,
        height_of_input_map=128, width_of_input_map=128,
        height_of_filter=3, width_of_filter=3,
        num_of_output_channels=16, convolution_stride=1,
        type="conv"
    )

    cost_conv_depth_seperable = UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=3,
        height_of_input_map=128, width_of_input_map=128,
        height_of_filter=3, width_of_filter=3,
        num_of_output_channels=16, convolution_stride=1,
        type="depth_seperable"
    )

    print("X")


def resnet_cost(i_c, o_c, stride, h_i, w_i, add_residual):
    cost_1 = UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=i_c,
        height_of_input_map=h_i, width_of_input_map=w_i,
        height_of_filter=1, width_of_filter=1,
        num_of_output_channels=o_c / 4, convolution_stride=1,
        type="conv"
    )
    cost_2 = UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=o_c / 4,
        height_of_input_map=h_i, width_of_input_map=h_i,
        height_of_filter=3, width_of_filter=3,
        num_of_output_channels=o_c / 4, convolution_stride=stride,
        type="depth_seperable"
    )

    cost_3 = UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=o_c / 4,
        height_of_input_map=h_i / stride, width_of_input_map=w_i / stride,
        height_of_filter=1, width_of_filter=1,
        num_of_output_channels=o_c, convolution_stride=1,
        type="conv"
    )

    cost = cost_1 + cost_2 + cost_3

    if add_residual:
        cost_4 = UtilityFuncs.calculate_mac_of_computation(
            num_of_input_channels=i_c,
            height_of_input_map=h_i, width_of_input_map=w_i,
            height_of_filter=1, width_of_filter=1,
            num_of_output_channels=o_c, convolution_stride=stride,
            type="conv"
        )
        cost += cost_4
    return cost


if __name__ == "__main__":
    cost = 0.0
    # Block 0
    cost += UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=3,
        height_of_input_map=32, width_of_input_map=32,
        height_of_filter=3, width_of_filter=3,
        num_of_output_channels=8, convolution_stride=1,
        type="conv"
    )
    cost += UtilityFuncs.calculate_mac_of_computation(
        num_of_input_channels=8,
        height_of_input_map=32, width_of_input_map=32,
        height_of_filter=1, width_of_filter=1,
        num_of_output_channels=32, convolution_stride=1,
        type="conv"
    )
    # Block 1
    cost += resnet_cost(i_c=32, o_c=32, stride=1, h_i=32, w_i=32, add_residual=True)
    for _ in range(2):
        cost += resnet_cost(i_c=32, o_c=32, stride=1, h_i=32, w_i=32, add_residual=False)

    # Block 2
    cost += resnet_cost(i_c=32, o_c=64, stride=2, h_i=32, w_i=32, add_residual=True)
    for _ in range(3):
        cost += resnet_cost(i_c=64, o_c=64, stride=1, h_i=16, w_i=16, add_residual=False)

    # Block 3
    cost += resnet_cost(i_c=64, o_c=256, stride=2, h_i=16, w_i=16, add_residual=True)
    for _ in range(3):
        cost += resnet_cost(i_c=256, o_c=256, stride=1, h_i=8, w_i=8, add_residual=False)

    # Block 4
    cost += resnet_cost(i_c=256, o_c=512, stride=1, h_i=8, w_i=8, add_residual=True)
    for _ in range(4):
        cost += resnet_cost(i_c=512, o_c=512, stride=1, h_i=8, w_i=8, add_residual=False)

    # Block 4
    for _ in range(4):
        cost += UtilityFuncs.calculate_mac_of_computation(
            num_of_input_channels=512,
            height_of_input_map=1, width_of_input_map=1,
            height_of_filter=1, width_of_filter=1,
            num_of_output_channels=100, convolution_stride=1,
            type="fc"
        )

    print("X")
