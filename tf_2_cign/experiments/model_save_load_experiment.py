import tensorflow as tf

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign
from tf_2_cign.utilities import Utilities

# Hyper-parameters
from tf_2_cign.fashion_net.fashion_cign_rl import FashionCignRl
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

input_dims = (28, 28, 1)
degree_list = [2, 2]
batch_size = 125
epoch_count = 100
decision_drop_probability = 0.0
drop_probability = 0.0
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
lambda_mac_cost = 0.1
q_net_params = [
    {
        "Conv_Filter": 1,
        "Conv_Strides": (1, 1),
        "Conv_Feature_Maps": 32,
        "Hidden_Layers": [32]
    },
    {
        "Conv_Filter": 1,
        "Conv_Strides": (1, 1),
        "Conv_Feature_Maps": 32,
        "Hidden_Layers": [64]
    }
]
warm_up_period = 25
rl_cign_iteration_period = 10

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    fashion_mnist = FashionMnist(batch_size=batch_size, validation_size=5000)
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=softmax_decay_initial,
                                                      decay_coefficient=softmax_decay_coefficient,
                                                      decay_period=softmax_decay_period,
                                                      decay_min_limit=softmax_decay_min_limit)
    with tf.device("GPU"):
        cign = FashionCignRl(valid_prediction_reward=valid_prediction_reward,
                             invalid_prediction_penalty=invalid_prediction_penalty,
                             include_ig_in_reward_calculations=True,
                             lambda_mac_cost=lambda_mac_cost,
                             q_net_params=q_net_params,
                             batch_size=batch_size,
                             input_dims=input_dims,
                             node_degrees=degree_list,
                             filter_counts=filter_counts,
                             kernel_sizes=kernel_sizes,
                             hidden_layers=hidden_layers,
                             decision_drop_probability=decision_drop_probability,
                             classification_drop_probability=drop_probability,
                             decision_wd=decision_wd,
                             classification_wd=classification_wd,
                             decision_dimensions=decision_dimensions,
                             class_count=10,
                             information_gain_balance_coeff=1.0,
                             softmax_decay_controller=softmax_decay_controller,
                             learning_rate_schedule=learning_rate_calculator,
                             decision_loss_coeff=1.0,
                             warm_up_period=warm_up_period,
                             cign_rl_train_period=rl_cign_iteration_period)

        cign2 = FashionCignRl(valid_prediction_reward=valid_prediction_reward,
                              invalid_prediction_penalty=invalid_prediction_penalty,
                              include_ig_in_reward_calculations=True,
                              lambda_mac_cost=lambda_mac_cost,
                              q_net_params=q_net_params,
                              batch_size=batch_size,
                              input_dims=input_dims,
                              node_degrees=degree_list,
                              filter_counts=filter_counts,
                              kernel_sizes=kernel_sizes,
                              hidden_layers=hidden_layers,
                              decision_drop_probability=decision_drop_probability,
                              classification_drop_probability=drop_probability,
                              decision_wd=decision_wd,
                              classification_wd=classification_wd,
                              decision_dimensions=decision_dimensions,
                              class_count=10,
                              information_gain_balance_coeff=1.0,
                              softmax_decay_controller=softmax_decay_controller,
                              learning_rate_schedule=learning_rate_calculator,
                              decision_loss_coeff=1.0,
                              warm_up_period=warm_up_period,
                              cign_rl_train_period=rl_cign_iteration_period)

        run_id = DbLogger.get_run_id()
        cign.init()
        explanation = cign.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        cign.train(run_id=run_id, dataset=fashion_mnist, epoch_count=epoch_count)
        cign.save_model(run_id=run_id)

        cign2.init()
        cign2.load_model(run_id=run_id)

        for train_X, train_y in fashion_mnist.trainDataTf:
            model_output_1 = cign.run_model(
                X=train_X,
                y=train_y,
                iteration=0,
                is_training=True,
                warm_up_period=False)

            model_output_2 = cign2.run_model(
                X=train_X,
                y=train_y,
                iteration=0,
                is_training=True,
                warm_up_period=False)

            are_outputs_equal = Utilities.compare_model_outputs(output_1=model_output_1, output_2=model_output_2)
            assert are_outputs_equal

        print("Test Passed!")