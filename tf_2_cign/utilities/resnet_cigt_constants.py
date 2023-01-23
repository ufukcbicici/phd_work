from math import ceil

from auxillary.parameters import DiscreteParameter
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm


class ResnetCigtConstants:
    # Standart Parameters
    input_dims = (32, 32, 3)
    class_count = 10
    batch_size = 256
    epoch_count = 125
    classification_wd = 0.0
    decision_wd = 0.0
    softmax_decay_initial = 25.0
    softmax_decay_coefficient = 0.9999
    softmax_decay_period = 1
    softmax_decay_min_limit = 0.1
    softmax_decay_controllers = {}
    information_gain_balance_coeff = 1.0
    decision_drop_probability = 0.0
    classification_drop_probability = 0.0
    batch_norm_type = "StandardBatchNormalization"
    apply_mask_to_batch_norm = False
    start_moving_averages_from_zero = False
    # assert batch_norm_type in {"StandardBatchNormalization",
    #                            "CigtBatchNormalization",
    #                            "CigtProbabilisticBatchNormalization"}
    bn_momentum = 0.9
    evaluation_period = 10
    measurement_start = 11
    decision_dimensions = [128, 128]
    initial_lr = 0.1
    iteration_count_per_epoch = ceil(50000 / batch_size) + 1 if 50000 % batch_size != 0 else 50000 / batch_size
    learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                 value=initial_lr,
                                                 schedule=[
                                                     (iteration_count_per_epoch * 150, initial_lr * 0.1),
                                                     (iteration_count_per_epoch * 250, initial_lr * 0.01)])
    decision_loss_coeff = 1.0
    decision_average_pooling_strides = [4, 4]
    optimizer_type = "SGD"
    decision_non_linearity = "Softmax"
    save_model = False
    warm_up_period = 85
    routing_strategy_name = "Approximate_Training"
    use_straight_through = True
    first_conv_kernel_size = 3
    first_conv_output_dim = 16
    first_conv_stride = 1
    resnet_config_list = [
        {"path_count": 1, "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]},
        {"path_count": 2, "layer_structure": [{"layer_count": 18, "feature_map_count": 32}]},
        {"path_count": 4, "layer_structure": [{"layer_count": 18, "feature_map_count": 64}]}]
    double_stride_layers = {18, 36}

    softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=softmax_decay_initial,
        decay_coefficient=softmax_decay_coefficient,
        decay_period=softmax_decay_period,
        decay_min_limit=softmax_decay_min_limit)

    # path_counts = [2, 4]

    # path_counts_with_leaf = [1]
    # path_counts_with_leaf.extend(path_counts)
    # # Format: [(b0, f0), (b1, f1), ..., (bn, fn)] for CIGT block i means
    # # There will b0 Resnet blocks with feature count f0 first, then b1 Resnet block with f1 features,
    # # finally bn blocks with fn features. b0 + b1 + ... + bn = Bi means there will be Bi Resnet blocks
    # # in the i.th CIGT block.
    # feature_counts_per_block = [
    #     [(18, 16)],
    #     [(18, 32)],
    #     [(18, 64)]]
    #
    # temp = []
    # for arr in feature_counts_per_block:
    #     temp2 = []
    #     for tpl in arr:
    #         temp2.extend(tpl[0] * [tpl[1]])
    #     temp.append(temp2)
    # feature_counts_per_block = temp
    #
    #
    #
    # block_parameters_list = []
    # # # Block Compositions, default settings first
    # # for block_id, path_count in enumerate(path_counts_with_leaf):
    # #     layer_composition_list = feature_counts_per_block[block_id]
    # #     block_parameters = []
    # #     for feature_block_id, tpl in enumerate(layer_composition_list):
    # #         layer_count = tpl[0]
    # #         feature_map_count = tpl[1]
    # #         for resnet_block_id in range(layer_count):
    #
    #
    #
    #
    #         #     if block_id > 0:
    #         #         in_dimension = feature_counts_per_block[block_id - 1][-1][-1]
    #         #         input_path_count =
    #         #     else:
    #         #         in_dimension = feature_map_count
    #         #     out_dimension = feature_map_count
    #         #
    #         #
    #         #
    #         #     block_params_object = {
    #         #         "in_dimension": in_dimension,
    #         #         "out_dimension": out_dimension,
    #         #         "input_path_count":
    #         #     }
    #         #
    #         #     #
    #         #     # in_dimension = block_params_object["in_dimension"]
    #         #     # # Number of feature maps exiting the block
    #         #     # out_dimension = block_params_object["out_dimension"]
    #         #     # # Number of routes entering the block
    #         #     # input_path_count = block_params_object["input_path_count"]
    #         #     # # Number of routes exiting the block
    #         #     # output_path_count = block_params_object["output_path_count"]
    #         #     # # Stride of the block's input convolution layer. When this is larger than 1, it means that we are going to
    #         #     # # apply dimension reduction to feature maps.
    #         #     # stride = block_params_object["stride"]
    #         #     #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # #     for basic_block_id in range(basic_block_count):
    # #
    # #
    # #
    # #
    # # block_id = 0
    # # for basic_block_id in range(18):
    # #     block_params_object = {
    # #         "in_dimension": 16,
    # #         "out_dimension": 16,
    # #         "input_path_count": path_counts_with_leaf[block_id]
    # #     }
    # #     # Number of feature maps entering the block
    # #     in_dimension = block_params_object["in_dimension"]
    # #     # Number of feature maps exiting the block
    # #     out_dimension = block_params_object["out_dimension"]
    # #     # Number of routes entering the block
    # #     input_path_count = block_params_object["input_path_count"]
    # #     # Number of routes exiting the block
    # #     output_path_count = block_params_object["output_path_count"]
    # #     # Stride of the block's input convolution layer. When this is larger than 1, it means that we are going to
    # #     # apply dimension reduction to feature maps.
    # #     stride = block_params_object["stride"]
    #
    # # # FashionNet parameters
    # # bn_momentum = 0.9
    # # filter_counts = [32, 64, 128]
    # # kernel_sizes = [5, 5, 1]
    # # hidden_layers = [512, 256]
    # # decision_dimensions = [128, 128]
    # # # node_build_funcs = [FashionCign.inner_func, FashionCign.inner_func, FashionCign.leaf_func]
    # # initial_lr = 0.01
    # # learning_rate_calculator = DiscreteParameter(name="lr_calculator",
    # #                                              value=initial_lr,
    # #                                              schedule=[(15000 + 12000, 0.005),
    # #                                                        (30000 + 12000, 0.0025),
    # #                                                        (40000 + 12000, 0.00025)])
    # #
    # # # CIGT Parameters
    # # path_counts = [2, 4]
    # #
    # # # Reinforcement learning routing parameters
    # # valid_prediction_reward = 1.0
    # # invalid_prediction_penalty = 0.0
    # # lambda_mac_cost = 0.5
    # # q_net_params = [
    # #     {
    # #         "Conv_Filter": 1,
    # #         "Conv_Strides": (1, 1),
    # #         "Conv_Feature_Maps": 32,
    # #         "Hidden_Layers": [64]
    # #     },
    # #     {
    # #         "Conv_Filter": 1,
    # #         "Conv_Strides": (1, 1),
    # #         "Conv_Feature_Maps": 32,
    # #         "Hidden_Layers": [32]
    # #     }
    # # ]
    # # warm_up_period = 25
    # # rl_cign_iteration_period = 10
    # # fine_tune_epoch_count = 25
    # # epsilon_decay_rate = 0.75
    # # epsilon_step = 1000
    # #
    # # experiment_factor = 10
