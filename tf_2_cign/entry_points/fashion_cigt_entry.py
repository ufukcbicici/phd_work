import tensorflow as tf
import numpy as np
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.cigt.cigt import Cigt
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities

# Hyper-parameters
from tf_2_cign.fashion_net.fashion_cign_rl import FashionCignRl
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size,
                                 validation_size=0)
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=FashionNetConstants.softmax_decay_initial,
                                                      decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
                                                      decay_period=FashionNetConstants.softmax_decay_period,
                                                      decay_min_limit=FashionNetConstants.softmax_decay_min_limit)

    with tf.device("GPU"):
        fashion_cigt = LenetCigt(batch_size=FashionNetConstants.batch_size,
                                 input_dims=FashionNetConstants.input_dims,
                                 filter_counts=FashionNetConstants.filter_counts,
                                 kernel_sizes=FashionNetConstants.kernel_sizes,
                                 hidden_layers=FashionNetConstants.hidden_layers,
                                 decision_drop_probability=FashionNetConstants.decision_drop_probability,
                                 classification_drop_probability=FashionNetConstants.classification_drop_probability,
                                 decision_wd=FashionNetConstants.decision_wd,
                                 classification_wd=FashionNetConstants.classification_wd,
                                 decision_dimensions=FashionNetConstants.decision_dimensions,
                                 class_count=10,
                                 information_gain_balance_coeff=FashionNetConstants.information_gain_balance_coeff,
                                 softmax_decay_controller=softmax_decay_controller,
                                 learning_rate_schedule=FashionNetConstants.learning_rate_calculator,
                                 decision_loss_coeff=1.0,
                                 path_counts=FashionNetConstants.path_counts,
                                 bn_momentum=FashionNetConstants.bn_momentum,
                                 warm_up_period=FashionNetConstants.warm_up_period,
                                 routing_strategy_name="Approximate_Training")
        fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf)

        # input_dims, class_count, path_counts, softmax_decay_controller, learning_rate_schedule,
        # decision_loss_coeff, routing_strategy_name, warm_up_period,


        # cigt = Cigt(input_dims=FashionNetConstants.input_dims,
        #             class_count=10,
        #             softmax_decay_controller=softmax_decay_controller,
        #             learning_rate_schedule=FashionNetConstants.learning_rate_calculator,
        #             decision_loss_coeff=1.0,
        #             path_counts=FashionNetConstants.path_counts,
        #             warm_up_period=FashionNetConstants.warm_up_period,
        #             routing_strategy_name="Approximate_Training")
        # print("X")
        # fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf)

        # fashion_cigt = Cigt(input_dims=FashionNetConstants.input_dims,
        #                     class_count=FashionNetConstants.class_count,
        #                     path_counts=path_counts,
        #                     blocks_list=None)

        # cign = FashionRlBinaryRouting(valid_prediction_reward=FashionNetConstants.valid_prediction_reward,
        #                               invalid_prediction_penalty=FashionNetConstants.invalid_prediction_penalty,
        #                               include_ig_in_reward_calculations=True,
        #                               lambda_mac_cost=FashionNetConstants.lambda_mac_cost,
        #                               q_net_params=FashionNetConstants.q_net_params,
        #                               batch_size=FashionNetConstants.batch_size,
        #                               input_dims=FashionNetConstants.input_dims,
        #                               node_degrees=FashionNetConstants.degree_list,
        #                               filter_counts=FashionNetConstants.filter_counts,
        #                               kernel_sizes=FashionNetConstants.kernel_sizes,
        #                               hidden_layers=FashionNetConstants.hidden_layers,
        #                               decision_drop_probability=FashionNetConstants.decision_drop_probability,
        #                               classification_drop_probability=FashionNetConstants.drop_probability,
        #                               decision_wd=FashionNetConstants.decision_wd,
        #                               classification_wd=FashionNetConstants.classification_wd,
        #                               decision_dimensions=FashionNetConstants.decision_dimensions,
        #                               class_count=10,
        #                               information_gain_balance_coeff=FashionNetConstants.information_gain_balance_coeff,
        #                               softmax_decay_controller=softmax_decay_controller,
        #                               learning_rate_schedule=FashionNetConstants.learning_rate_calculator,
        #                               decision_loss_coeff=1.0,
        #                               warm_up_period=FashionNetConstants.warm_up_period,
        #                               cign_rl_train_period=FashionNetConstants.rl_cign_iteration_period,
        #                               q_net_coeff=1.0,
        #                               epsilon_decay_rate=FashionNetConstants.epsilon_decay_rate,
        #                               epsilon_step=FashionNetConstants.epsilon_step,
        #                               reward_type="Zero Rewards")
        #
        # run_id = DbLogger.get_run_id()
        # cign.init()
        # explanation = cign.get_explanation_string()
        # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        #
        # cign.train_end_to_end(run_id=run_id,
        #                       dataset=fashion_mnist,
        #                       epoch_count=125,
        #                       q_net_epoch_count=250,
        #                       fine_tune_epoch_count=25,
        #                       warm_up_epoch_count=25,
        #                       q_net_train_start_epoch=25,
        #                       q_net_train_period=25)

        # Utilities.pickle_save_to_file(path=)

        # cign.load_model(run_id=3172, epoch_id=99)
        # cign.measure_performance(dataset=fashion_mnist, run_id=run_id, only_use_ig_routing=True)
        # cign.calculate_ideal_accuracy(dataset=fashion_mnist.testDataTf)
        #
        # autoencoders = cign.load_autoencoders(dataset=fashion_mnist,
        #                                       run_id_list=[3247, 3248],
        #                                       epoch_list=[378, 261])
        # for i in range(1000):
        #     delta = i * 0.0001
        #     print("{0}-{1}".format(0.051-2.0*delta, 0.320-delta))
        #     cign.calculate_accuracy_with_anomaly_detectors(dataset=fashion_mnist.testDataTf,
        #                                                    anomaly_detectors=autoencoders,
        #                                                    selection_thresholds=[0.051-2.0*delta, 0.320-delta])
        # print("X")

        # cign.train(run_id=run_id,
        #            dataset=fashion_mnist,
        #            epoch_count=100,
        #            q_net_epoch_count=250,
        #            fine_tune_epoch_count=25,
        #            warm_up_epoch_count=25,
        #            q_net_train_start_epoch=25,
        #            q_net_train_period=25)

        # cign.train_using_q_nets_as_post_processing(run_id=run_id,
        #                                            dataset=fashion_mnist,
        #                                            epoch_count=100,
        #                                            q_net_epoch_count=250,
        #                                            fine_tune_epoch_count=25,
        #                                            warm_up_epoch_count=25,
        #                                            q_net_train_start_epoch=25,
        #                                            q_net_train_period=25)

        # cign.train_q_nets_as_anomaly_detectors(run_id=3172,
        #                                        dataset=fashion_mnist,
        #                                        q_net_epoch_count=1000)

        # print("X")

        # cign.load_model(run_id=3094, epoch_id=25)
        # cign.train_q_nets_with_full_net(dataset=fashion_mnist, q_net_epoch_count=250)
