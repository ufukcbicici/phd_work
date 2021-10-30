from auxillary.parameters import DiscreteParameter


class FashionNetConstants:
    # Fashion_net_constants
    input_dims = (28, 28, 1)
    degree_list = [2, 2]
    batch_size = 125
    epoch_count = 110
    decision_drop_probability = 0.0
    drop_probability = 0.15
    classification_wd = 0.0
    decision_wd = 0.0
    softmax_decay_initial = 25.0
    softmax_decay_coefficient = 0.9999
    softmax_decay_period = 2
    softmax_decay_min_limit = 1.0
    softmax_decay_controllers = {}

    # FashionNet parameters
    filter_counts = [32, 32, 32]
    kernel_sizes = [5, 5, 1]
    hidden_layers = [128, 64]
    decision_dimensions = [128, 128]
    # node_build_funcs = [FashionCign.inner_func, FashionCign.inner_func, FashionCign.leaf_func]
    initial_lr = 0.01
    learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                 value=initial_lr,
                                                 schedule=[(15000, 0.005),
                                                           (30000, 0.0025),
                                                           (40000, 0.00025)])

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
    epsilon_step = 5000
