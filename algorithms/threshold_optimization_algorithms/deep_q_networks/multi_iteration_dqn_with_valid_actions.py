import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from algorithms.threshold_optimization_algorithms.deep_q_networks.multi_iteration_deep_q_learning import \
    MultiIterationDQN
from algorithms.threshold_optimization_algorithms.deep_q_networks.multi_iteration_dqn_with_regression import \
    MultiIterationDQNRegression
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class MultiIterationDQNRegressionValidActions(MultiIterationDQNRegression):
    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost):
        self.rewardMatrix = None
        self.lossMatrix = None
        super().__init__(routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                         lambda_mac_cost)
        self.lrBoundaries = [5000, 10000, 20000]
        self.lrValues = [0.1, 0.01, 0.001, 0.0001]
        self.learningRate = tf.train.piecewise_constant(self.globalStep, self.lrBoundaries, self.lrValues)
        self.build_loss()

    def build_loss(self):
        # Get selected q values; build the regression loss: MSE or Huber between Last layer Q outputs and the reward
        self.rewardMatrix = tf.placeholder(dtype=tf.float32, shape=[None, self.actionSpaces[-1].shape[0]],
                                           name="reward_matrix")
        self.lossMatrix = tf.square(self.qFunction - self.rewardMatrix)
        self.lossValue = tf.reduce_mean(self.lossMatrix)
        self.get_l2_loss()
        self.totalLoss = self.lossValue + self.l2Loss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
        # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).\
        #     minimize(self.totalLoss, global_step=self.globalStep)

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
        kwargs["lrValues"] = self.lrValues
        kwargs["lrBoundaries"] = self.lrBoundaries
        run_id = self.log_meta_data(kwargs=kwargs)
        losses = []
        # Test the accuracy evaluations
        self.evaluate(run_id=run_id, episode_id=-1, discount_factor=discount_factor)
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
                self.evaluate(run_id=run_id, episode_id=episode_id, discount_factor=discount_factor)
