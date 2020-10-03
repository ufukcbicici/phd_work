import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from algorithms.threshold_optimization_algorithms.deep_q_networks.multi_iteration_deep_q_learning import \
    MultiIterationDQN
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class MultiIterationDQNRegression(MultiIterationDQN):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost):
        self.rewardMatrix = None
        self.lossMatrix = None
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                         lambda_mac_cost)

    def build_q_function(self):
        level = self.get_max_trajectory_length() - 1
        if self.qLearningFunc == "cnn":
            nodes_at_level = self.network.orderedNodesPerLevel[level]
            shapes_list = [self.stateFeatures[iteration][node.index].shape
                           for iteration in self.routingDataset.iterations for node in nodes_at_level]
            assert len(set(shapes_list)) == 1
            entry_shape = list(shapes_list[0])
            entry_shape[0] = None
            entry_shape[-1] = len(nodes_at_level) * entry_shape[-1]
            self.stateInput = tf.placeholder(dtype=tf.float32, shape=entry_shape, name="state_inputs")
            self.build_cnn_q_network()
        # Get selected q values; build the regression loss: MSE or Huber between Last layer Q outputs and the reward
        self.rewardMatrix = tf.placeholder(dtype=tf.float32, shape=[None, self.actionSpaces[-1].shape[0]],
                                           name="reward_matrix")
        self.lossMatrix = tf.square(self.qFunction - self.rewardMatrix)
        self.lossValue = tf.reduce_mean(self.lossMatrix)
        self.get_l2_loss()
        self.totalLoss = self.lossValue + self.l2Loss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)

    def train(self, level, **kwargs):
        sample_count = kwargs["sample_count"]
        episode_count = kwargs["episode_count"]
        discount_factor = kwargs["discount_factor"]
        epsilon_discount_factor = kwargs["epsilon_discount_factor"]
        learning_rate = kwargs["learning_rate"]
        epsilon = 1.0
        if level != self.get_max_trajectory_length() - 1:
            raise NotImplementedError()
        self.session.run(tf.global_variables_initializer())
        # If we use only information gain for routing (ML: Maximum likelihood routing)
        whole_data_ml_accuracy = self.get_max_likelihood_accuracy(iterations=self.routingDataset.iterations,
                                                                  sample_indices=np.arange(
                                                                      self.routingDataset.labelList.shape[0]))
        training_ml_accuracy = self.get_max_likelihood_accuracy(iterations=self.routingDataset.iterations,
                                                                sample_indices=self.routingDataset.trainingIndices)
        test_ml_accuracy = self.get_max_likelihood_accuracy(iterations=self.routingDataset.testIterations,
                                                            sample_indices=self.routingDataset.testIndices)
        # Fill the explanation string for the experiment
        kwargs["whole_data_ml_accuracy"] = whole_data_ml_accuracy
        kwargs["training_ml_accuracy"] = training_ml_accuracy
        kwargs["test_ml_accuracy"] = test_ml_accuracy
        kwargs["invalid_action_penalty"] = MultiIterationDQN.invalid_action_penalty
        kwargs["valid_prediction_reward"] = MultiIterationDQN.valid_prediction_reward
        kwargs["invalid_prediction_penalty"] = MultiIterationDQN.invalid_prediction_penalty
        kwargs["INCLUDE_IG_IN_REWARD_CALCULATIONS"] = MultiIterationDQN.INCLUDE_IG_IN_REWARD_CALCULATIONS
        kwargs["CONV_FEATURES"] = MultiIterationDQN.CONV_FEATURES
        kwargs["HIDDEN_LAYERS"] = MultiIterationDQN.HIDDEN_LAYERS
        kwargs["FILTER_SIZES"] = MultiIterationDQN.FILTER_SIZES
        kwargs["STRIDES"] = MultiIterationDQN.STRIDES
        kwargs["MAX_POOL"] = MultiIterationDQN.MAX_POOL
        experiment_id = DbLogger.get_run_id()
        explanation_string = "DQN Experiment. RunID:{0}\n".format(experiment_id)
        for k, v in kwargs.items():
            explanation_string += "{0}:{1}\n".format(k, v)
        print("Whole Data ML Accuracy{0}".format(whole_data_ml_accuracy))
        print("Training Set ML Accuracy:{0}".format(training_ml_accuracy))
        print("Test Set ML Accuracy:{0}".format(test_ml_accuracy))
        DbLogger.write_into_table(rows=[(experiment_id, explanation_string)], table=DbLogger.runMetaData, col_count=2)
        losses = []
        # Test the accuracy evaluations
        self.evaluate(run_id=experiment_id, episode_id=-1, discount_factor=discount_factor)
        for episode_id in range(episode_count):
            print("Episode:{0}".format(episode_id))
            sample_ids = np.random.choice(self.routingDataset.trainingIndices, sample_count, replace=True)
            iterations = np.random.choice(self.routingDataset.iterations, sample_count, replace=True)
            actions_t_minus_1 = np.random.choice(self.actionSpaces[level - 1].shape[0], sample_count, replace=True)
            rewards_matrix = self.get_rewards(samples=sample_ids, iterations=iterations,
                                              action_ids_t_minus_1=actions_t_minus_1, action_ids_t=None, level=level)
            state_features = self.get_state_features(samples=sample_ids,
                                                     iterations=iterations,
                                                     action_ids_t_minus_1=actions_t_minus_1, level=level)
            results = self.session.run([self.totalLoss, self.lossMatrix, self.lossValue, self.optimizer],
                                       feed_dict={self.stateCount: sample_count,
                                                  self.stateInput: state_features,
                                                  self.rewardMatrix: rewards_matrix,
                                                  self.l2LambdaTf: 0.0})
            total_loss = results[0]
            losses.append(total_loss)
            if len(losses) % 10 == 0:
                print("Episode:{0} MSE:{1}".format(episode_id, np.mean(np.array(losses))))
                losses = []
            if (episode_id + 1) % 200 == 0:
                self.evaluate(run_id=experiment_id, episode_id=episode_id, discount_factor=discount_factor)
