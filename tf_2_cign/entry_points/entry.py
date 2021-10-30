import tensorflow as tf
import numpy as np
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign

# Hyper-parameters
from tf_2_cign.fashion_net.fashion_cign_rl import FashionCignRl
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size,
                                 validation_size=5000,
                                 validation_source="training")
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=FashionNetConstants.softmax_decay_initial,
                                                      decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
                                                      decay_period=FashionNetConstants.softmax_decay_period,
                                                      decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
    with tf.device("GPU"):
        # cign = FashionCign(batch_size=batch_size,
        #                    input_dims=input_dims,
        #                    node_degrees=degree_list,
        #                    filter_counts=filter_counts,
        #                    kernel_sizes=kernel_sizes,
        #                    hidden_layers=hidden_layers,
        #                    decision_drop_probability=decision_drop_probability,
        #                    classification_drop_probability=drop_probability,
        #                    decision_wd=decision_wd,
        #                    classification_wd=classification_wd,
        #                    decision_dimensions=decision_dimensions,
        #                    class_count=10,
        #                    information_gain_balance_coeff=1.0,
        #                    softmax_decay_controller=softmax_decay_controller,
        #                    learning_rate_schedule=learning_rate_calculator,
        #                    decision_loss_coeff=1.0)

        # cign = FashionCignRl(valid_prediction_reward=valid_prediction_reward,
        #                      invalid_prediction_penalty=invalid_prediction_penalty,
        #                      include_ig_in_reward_calculations=True,
        #                      lambda_mac_cost=lambda_mac_cost,
        #                      q_net_params=q_net_params,
        #                      batch_size=batch_size,
        #                      input_dims=input_dims,
        #                      node_degrees=degree_list,
        #                      filter_counts=filter_counts,
        #                      kernel_sizes=kernel_sizes,
        #                      hidden_layers=hidden_layers,
        #                      decision_drop_probability=decision_drop_probability,
        #                      classification_drop_probability=drop_probability,
        #                      decision_wd=decision_wd,
        #                      classification_wd=classification_wd,
        #                      decision_dimensions=decision_dimensions,
        #                      class_count=10,
        #                      information_gain_balance_coeff=1.0,
        #                      softmax_decay_controller=softmax_decay_controller,
        #                      learning_rate_schedule=learning_rate_calculator,
        #                      decision_loss_coeff=1.0,
        #                      warm_up_period=warm_up_period,
        #                      cign_rl_train_period=rl_cign_iteration_period)

        cign = FashionCignRl(valid_prediction_reward=FashionNetConstants.valid_prediction_reward,
                             invalid_prediction_penalty=FashionNetConstants.invalid_prediction_penalty,
                             include_ig_in_reward_calculations=True,
                             lambda_mac_cost=FashionNetConstants.lambda_mac_cost,
                             q_net_params=FashionNetConstants.q_net_params,
                             batch_size=FashionNetConstants.batch_size,
                             input_dims=FashionNetConstants.input_dims,
                             node_degrees=FashionNetConstants.degree_list,
                             filter_counts=FashionNetConstants.filter_counts,
                             kernel_sizes=FashionNetConstants.kernel_sizes,
                             hidden_layers=FashionNetConstants.hidden_layers,
                             decision_drop_probability=FashionNetConstants.decision_drop_probability,
                             classification_drop_probability=FashionNetConstants.drop_probability,
                             decision_wd=FashionNetConstants.decision_wd,
                             classification_wd=FashionNetConstants.classification_wd,
                             decision_dimensions=FashionNetConstants.decision_dimensions,
                             class_count=10,
                             information_gain_balance_coeff=1.0,
                             softmax_decay_controller=softmax_decay_controller,
                             learning_rate_schedule=FashionNetConstants.learning_rate_calculator,
                             decision_loss_coeff=1.0,
                             warm_up_period=FashionNetConstants.warm_up_period,
                             cign_rl_train_period=FashionNetConstants.rl_cign_iteration_period,
                             q_net_coeff=1.0)

        run_id = DbLogger.get_run_id()
        cign.init()
        explanation = cign.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        # cign.load_model(run_id=2965)
        # cign.train_q_nets_with_full_net(dataset=fashion_mnist, q_net_epoch_count=250)




        # cign.calculate_optimal_q_values(dataset=fashion_mnist.validationDataTf, batch_size=batch_size)
        cign.train(run_id=run_id,
                   dataset=fashion_mnist,
                   epoch_count=100,
                   q_net_epoch_count=250,
                   fine_tune_epoch_count=25)

        # cign.train_without_val_set(run_id=run_id, dataset=fashion_mnist, epoch_count=100)

        # RL Routing experiments
        # # cign.load_model(run_id=2687)
        # cign.load_model(run_id=2766)
        # # cign.measure_performance(dataset=fashion_mnist, run_id=run_id, iteration=0, epoch_id=0, times_list=[])
        # # cign.train_q_nets(dataset=fashion_mnist, q_net_epoch_count=250)
        # # cign.measure_performance(dataset=fashion_mnist, run_id=run_id, iteration=0, epoch_id=0, times_list=[])
        # # cign.save_model(run_id=run_id)
        #
        # # cign.load_model(run_id=2723)
        # #
        # q_learning_dataset_val = \
        #     cign.calculate_optimal_q_values(dataset=fashion_mnist.validationDataTf,
        #                                     batch_size=cign.batchSizeNonTensor, shuffle_data=False)
        # q_learning_dataset_train = \
        #     cign.calculate_optimal_q_values(dataset=fashion_mnist.trainDataTf,
        #                                     batch_size=cign.batchSizeNonTensor, shuffle_data=False)
        # q_learning_dataset_test = \
        #     cign.calculate_optimal_q_values(dataset=fashion_mnist.testDataTf,
        #                                     batch_size=cign.batchSizeNonTensor, shuffle_data=False)
        #
        # # q_predicted_tables, last_q_table_no_penalties_predicted, y_pred_1 = cign.eval_q_nets(
        # #     dataset=q_learning_dataset_val)
        #
        # cign.measure_performance(dataset=fashion_mnist, run_id=run_id, iteration=-1, epoch_id=-1,
        #                          times_list=[])
