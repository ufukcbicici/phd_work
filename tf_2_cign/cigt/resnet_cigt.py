from tf_2_cign.cigt.cigt import Cigt


class ResnetCigt(Cigt):
    def __init__(self, run_id, batch_size, input_dims,
                 decision_drop_probability, classification_drop_probability, decision_wd, classification_wd,
                 evaluation_period, measurement_start, decision_dimensions, class_count, information_gain_balance_coeff,
                 softmax_decay_controller, learning_rate_schedule, optimizer_type, decision_non_linearity,
                 decision_loss_coeff, path_counts, bn_momentum, warm_up_period, routing_strategy_name,
                 use_straight_through, save_model, model_definition, *args, **kwargs):
        super().__init__(run_id, batch_size, input_dims, class_count, path_counts, softmax_decay_controller,
                         learning_rate_schedule, optimizer_type, decision_non_linearity, decision_loss_coeff,
                         routing_strategy_name, use_straight_through, warm_up_period, decision_drop_probability,
                         classification_drop_probability, decision_wd, classification_wd, evaluation_period,
                         measurement_start, save_model, model_definition, *args, **kwargs)