from auxillary.parameters import DiscreteParameter


class FashionNetConstants:
    # Fashion_net_constants
    input_dims = (28, 28, 1)
    degree_list = [2, 2]
    class_count = 10
    batch_size = 125
    epoch_count = 125
    decision_drop_probability = 0.0
    classification_drop_probability = 0.0
    classification_wd = 0.0
    decision_wd = 0.0
    softmax_decay_initial = 25.0
    softmax_decay_coefficient = 0.9999
    softmax_decay_period = 1
    softmax_decay_min_limit = 0.1
    softmax_decay_controllers = {}
    information_gain_balance_coeff = 1.0

    # FashionNet parameters
    bn_momentum = 0.9
    filter_counts = [32, 64, 128]
    kernel_sizes = [5, 5, 1]
    hidden_layers = [512, 256]
    decision_dimensions = [128, 128]
    # node_build_funcs = [FashionCign.inner_func, FashionCign.inner_func, FashionCign.leaf_func]
    initial_lr = 0.01
    learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                 value=initial_lr,
                                                 schedule=[(15000 + 12000, 0.005),
                                                           (30000 + 12000, 0.0025),
                                                           (40000 + 12000, 0.00025)])

    # CIGT Parameters
    path_counts = [2, 4]

    # Reinforcement learning routing parameters
    valid_prediction_reward = 1.0
    invalid_prediction_penalty = 0.0
    lambda_mac_cost = 0.5
    q_net_params = [
        {
            "Conv_Filter": 1,
            "Conv_Strides": (1, 1),
            "Conv_Feature_Maps": 32,
            "Hidden_Layers": [64]
        },
        {
            "Conv_Filter": 1,
            "Conv_Strides": (1, 1),
            "Conv_Feature_Maps": 32,
            "Hidden_Layers": [32]
        }
    ]
    warm_up_period = 25
    rl_cign_iteration_period = 10
    fine_tune_epoch_count = 25
    epsilon_decay_rate = 0.75
    epsilon_step = 1000

    experiment_factor = 10
