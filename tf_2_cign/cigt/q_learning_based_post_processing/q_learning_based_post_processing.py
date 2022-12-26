import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.data_classes.multipath_routing_info2 import MultipathCombinationInfo2
from tf_2_cign.cigt.q_learning_based_post_processing.lstm_based_q_model import LstmBasedQModel
from tf_2_cign.utilities.utilities import Utilities


class QLearningRoutingOptimizer(object):
    intermediate_outputs_path = os.path.join(os.path.dirname(__file__), "..", "intermediate_outputs")

    def __init__(self, run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader,
                 model_id, val_ratio, max_test_val_diff, random_seed):
        self.runId = run_id
        self.modelId = model_id
        self.valRatio = val_ratio
        self.maxTestValDiff = max_test_val_diff
        self.numOfEpochs = num_of_epochs
        self.randomSeed = random_seed
        self.accuracyWeight = accuracy_weight
        self.macWeight = mac_weight
        self.lr = 0.001
        self.learningRateSchedule = DiscreteParameter(name="lr_calculator",
                                                      value=self.lr,
                                                      schedule=[(15000 + 12000, (1.0 / 2.0) * self.lr),
                                                                (30000 + 12000, (1.0 / 4.0) * self.lr),
                                                                (40000 + 12000, (1.0 / 40.0) * self.lr)])
        self.optimizer = self.get_optimizer()
        self.modelLoader = model_loader
        self.model, self.dataset = self.modelLoader.get_model(model_id=self.modelId)
        self.pathCounts = list(self.model.pathCounts)
        self.totalSampleCount, self.valIndices, self.testIndices = None, None, None
        self.multiPathInfoObject = self.load_multipath_info()
        # self.softmaxTemperatureOptimizer = SoftmaxTemperatureOptimizer(multi_path_object=self.multiPathInfoObject)
        # self.softmaxTemperatureOptimizer.plot_entropy_histogram_with_temperature(temperature=1.0, block_id=0)
        self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()
        self.qTables = None

    # Load routing information for the particular model
    def load_multipath_info(self):
        # object_path = os.path.join(object_folder_path, "multipath_info_object.pkl")
        multipath_info_object = MultipathCombinationInfo2(batch_size=self.model.batchSize,
                                                          path_counts=self.pathCounts)
        multipath_info_object.generate_routing_info(
            cigt=self.model,
            dataset=self.dataset.testDataTf,
            apply_temperature_optimization_to_entropies=False,
            apply_temperature_optimization_to_routing_probabilities=False)
        multipath_info_object.assert_routing_validity(cigt=self.model)
        multipath_info_object.assess_accuracy()
        return multipath_info_object

    def prepare_val_test_sets(self):
        total_sample_count = set()
        for ll in self.multiPathInfoObject.combinations_routing_probabilities_dict.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for ll in self.multiPathInfoObject.combinations_routing_entropies_dict.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for arr in self.multiPathInfoObject.combinations_y_hat_dict.values():
            total_sample_count.add(arr.shape[0])
        for arr in self.multiPathInfoObject.combinations_y_dict.values():
            total_sample_count.add(arr.shape[0])
        assert len(total_sample_count) == 1
        total_sample_count = list(total_sample_count)[0]
        val_sample_count = int(total_sample_count * self.valRatio)
        indices = np.arange(total_sample_count)
        np.random.seed(self.randomSeed)
        val_indices, test_indices = train_test_split(indices, train_size=val_sample_count)
        return total_sample_count, val_indices, test_indices

    def get_last_block_routing_decisions(self, q_choice_combination):
        # Determine the corresponding routing choices for every sample
        path_selections = np.arange(self.totalSampleCount)[:, np.newaxis]
        for block_id, block_choice in enumerate(q_choice_combination):
            path_indices = tuple([path_selections[:, idx] for idx in range(path_selections.shape[1])])
            routing_probabilities = \
                self.multiPathInfoObject.past_decisions_routing_probabilities_list[block_id][path_indices]
            # Select the possible routing options
            # If block_choice = 0 -> ig selection
            ig_routings = Utilities.one_hot_numpy(arr=routing_probabilities)
            # If block_choice = 1 -> all paths
            all_paths_routings = np.ones(shape=ig_routings.shape, dtype=np.int32)
            selected_paths_for_block = np.where(block_choice, all_paths_routings, ig_routings)
            path_selections = np.concatenate([path_selections[:, :-1],
                                              selected_paths_for_block,
                                              np.arange(self.totalSampleCount)[:, np.newaxis]], axis=1)
        # Accuracy vector of selections
        path_indices = tuple([path_selections[:, idx] for idx in range(path_selections.shape[1])])
        validity_vector = self.multiPathInfoObject.past_decisions_validity_array[path_indices]
        mac_vector = self.multiPathInfoObject.past_decisions_mac_array[path_indices[:-1]]
        q_values = self.accuracyWeight * validity_vector.astype(np.float) + self.macWeight * mac_vector
        return q_values

    def prepare_q_tables(self):
        self.qTables = [np.zeros(shape=1)] * (len(self.model.pathCounts) - 1)
        for block_id in range(len(self.model.pathCounts) - 2, -1, -1):
            # Last block
            if block_id == len(self.model.pathCounts) - 2:
                choice_count = block_id + 1
                q_table_shape = np.concatenate([
                    np.array([self.totalSampleCount], dtype=np.int32),
                    np.array([2 for _ in range(choice_count)], dtype=np.int32)]).astype(dtype=np.int32)
                q_table = np.zeros(shape=q_table_shape, dtype=np.float32)
                choice_combinations = Utilities.get_cartesian_product(
                    list_of_lists=[[0, 1] for _ in range(choice_count)])
                transpose_axes = (*[d_ for d_ in range(1, choice_count + 1)], 0)
                q_table_transpose = np.transpose(q_table, axes=transpose_axes)
                for choice_combination in choice_combinations:
                    q_values = self.get_last_block_routing_decisions(q_choice_combination=choice_combination)
                    # q_table[choice_combination] = q_values
                    q_table[(slice(None), *choice_combination)] = q_values
                    assert np.array_equal(q_table_transpose[choice_combination], q_values)
                self.qTables[block_id] = q_table
            else:
                q_table = np.max(self.qTables[block_id + 1], axis=-1)
                self.qTables[block_id] = q_table
            print(block_id)

    def calibrate_test_and_val_sets(self, load_prev_indices=True):
        file_name = "test_val_indices.sav"
        if load_prev_indices and os.path.isfile(file_name):
            dict_loaded = Utilities.pickle_load_from_file(path=file_name)
            self.testIndices = dict_loaded["test_indices"]
            self.valIndices = dict_loaded["val_indices"]
            test_accuracy = self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.testIndices)
            val_accuracy = self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.valIndices)
            print("test_accuracy:{0}".format(test_accuracy))
            print("val_accuracy:{0}".format(val_accuracy))
        else:
            num_of_trials = 0
            while True:
                print("Num of Trials:{0}".format(num_of_trials))
                test_accuracy = self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.testIndices)
                val_accuracy = self.multiPathInfoObject.get_default_accuracy(cigt=self.model, indices=self.valIndices)
                print("test_accuracy:{0}".format(test_accuracy))
                print("val_accuracy:{0}".format(val_accuracy))
                if abs(test_accuracy - val_accuracy) <= self.maxTestValDiff:
                    Utilities.pickle_save_to_file(path=file_name, file_content={"test_indices": self.testIndices,
                                                                                "val_indices": self.valIndices})
                    print("Suitable test and validation indices have been found with {0} trials.".format(num_of_trials))
                    break
                num_of_trials += 1
                self.valIndices, self.testIndices = train_test_split(np.arange(self.totalSampleCount),
                                                                     train_size=len(self.valIndices))

    def get_selected_q_values(self, block_id, sample_indices, actions):
        assert sample_indices.shape[0] == actions.shape[0]
        # assert len(sample_indices.shape) == len(actions.shape)
        actions_prior = actions[:, :-1]
        idx = np.concatenate([sample_indices[:, np.newaxis], actions_prior], axis=1)
        # if actions.shape[1] == 1:
        #     # actions_prior = actions[:, :-1]
        #     idx = sample_indices[:, np.newaxis]
        # else:
        #     actions_prior = actions[:, :-1]
        #     idx = np.concatenate([sample_indices, actions_prior], axis=1)
        idx_tuple = Utilities.convert_trajectory_array_to_indices(trajectory_array=idx)
        q_vals = self.qTables[block_id][idx_tuple]
        return q_vals

    def convert_actions_to_routes(self, path_indices, actions_arr, block_id):
        routing_probabilities = \
            self.multiPathInfoObject.past_decisions_routing_probabilities_list[block_id][path_indices]
        # Select the possible routing options
        # If block_choice = 0 -> ig selection
        ig_routings = Utilities.one_hot_numpy(arr=routing_probabilities)
        # If block_choice = 1 -> all paths
        all_routings = np.ones_like(ig_routings)
        routes_selected = np.where(actions_arr[:, np.newaxis], all_routings, ig_routings)
        return routes_selected

    def get_optimizer(self):
        boundaries = [tpl[0] for tpl in self.learningRateSchedule.schedule]
        values = [self.learningRateSchedule.initialValue]
        values.extend([tpl[1] for tpl in self.learningRateSchedule.schedule])
        learning_rate_scheduler_tf = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate_scheduler_tf, momentum=0.9)
        # if self.optimizerType == "SGD":
        #     optimizer = tf.keras.optimizers.SGD(
        #         learning_rate=learning_rate_scheduler_tf, momentum=0.9)
        # elif self.optimizer == "Adam":
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler_tf)
        # else:
        #     raise NotImplementedError()
        return optimizer

    def get_inputs_outputs_from_rnn(self, sample_indices, actions):
        path_selections = sample_indices[:, np.newaxis]
        step_count = len(self.model.pathCounts) - 1
        features = self.multiPathInfoObject.past_decisions_h_features_list
        path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
        # h_t = features[0][path_indices]
        inputs = []
        targets = []
        for t in range(step_count):
            # Inputs for step t
            h_t = features[t][path_indices]
            inputs.append(h_t)
            # The actions we have sampled for step t, so far.
            a_1t = actions[:, :(t + 1)]
            # Current actions
            a_t = actions[:, t]
            # The output (Q-Table) for step t. Block indices start from left, go to the right.
            q_vals = self.get_selected_q_values(block_id=t, actions=a_1t, sample_indices=sample_indices)
            targets.append(q_vals)
            routes_selected = self.convert_actions_to_routes(path_indices=path_indices,
                                                             block_id=t,
                                                             actions_arr=a_t)
            path_selections = np.concatenate([path_selections[:, :-1],
                                              routes_selected,
                                              path_selections[:, -1][:, np.newaxis]], axis=1)
            path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
        q_tables_ground_truth = np.stack(targets, axis=1)
        q_tables_ground_truth = tf.convert_to_tensor(q_tables_ground_truth)
        return inputs, q_tables_ground_truth

    def train(self, epoch_count, batch_size, input_dimension, lstm_layer_dimensions,
              dropout_ratio):
        # Create the lstm model
        lstm_q_model = LstmBasedQModel(path_counts=self.model.pathCounts,
                                       input_dimension=input_dimension,
                                       lstm_layer_dimensions=lstm_layer_dimensions,
                                       dropout_ratio=dropout_ratio)
        step_count = len(self.model.pathCounts) - 1
        # Create the datasets
        datasets = {}
        for dataset_type, indices in [("validation", self.valIndices), ("test", self.testIndices)]:
            dataset_tf = tf.data.Dataset.from_tensor_slices((indices,)).shuffle(1000).batch(batch_size)
            datasets[dataset_type] = dataset_tf

        features = self.multiPathInfoObject.past_decisions_h_features_list
        for epoch_id in range(epoch_count):
            for tpl in datasets["validation"]:
                train_idx = tpl[0].numpy()
                path_selections = train_idx[:, np.newaxis]
                q_table_selections = train_idx[:, np.newaxis]
                # Step 1: Sample trajectories
                actions = np.random.randint(low=0, high=2, size=(train_idx.shape[0], step_count))

                # Step 2: Create network inputs and outputs; raw routing features and optimal q_tables
                inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=train_idx,
                                                                                 actions=actions)
                # Step 3: Transform h outputs from each block into Q table outputs.
                with tf.GradientTape() as tape:
                    q_tables_prediction = lstm_q_model(inputs, training=True)
                    q_tables_prediction = tf.stack(q_tables_prediction, axis=1)
                    mse_tensor = tf.square(q_tables_ground_truth - q_tables_prediction)
                    mse_loss = tf.reduce_mean(mse_tensor)
                    # mse_loss = tf.losses.mean_squared_error(q_tables_ground_truth, q_tables_prediction)
                print("mse_loss:{0}".format(mse_loss))
                grads = tape.gradient(mse_loss, lstm_q_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, lstm_q_model.trainable_variables))

            print("Validation Results")
            self.evaluate(lstm_q_model=lstm_q_model, dataset=datasets["validation"])
            print("Test Results")
            self.evaluate(lstm_q_model=lstm_q_model, dataset=datasets["test"])

    def evaluate(self, lstm_q_model, dataset):
        step_count = len(self.model.pathCounts) - 1
        validity_vectors = []
        mac_vectors = []
        for tpl in dataset:
            sample_indices = tpl[0].numpy()
            actions = np.zeros(shape=(sample_indices.shape[0], step_count), dtype=np.int32)
            path_selections = sample_indices[:, np.newaxis]
            path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
            # A very suboptimal algorithm:
            # Given a batch of samples, we load an actions array of the size BxS, where B is the batch size,
            # S is the step count, with zeros.
            # At the start, where t=0, we use the initial actions array.
            # The q_table output at t=0, will act as our guidance for the step t=1. We pick a_0 = argmax(q_0, axis=1).
            # Then load actions[:, 0] <- a_0. This means we are going to select s_1 according to the actions we have
            # just sampled. Then we run our inference again. The q_table output at t=1 will now show q_1, with respect
            # to actions at a_0. Then we pick a_1 = argmax(q_1, axis=1).
            # Without loss of generality, if we assume that we have 2 time steps, with a_1 is determined, we can pick
            # the optimal selected path from the start to end and then calculate the accuracy and MAC burden.

            for t in range(step_count):
                inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=sample_indices,
                                                                                 actions=actions)
                q_tables_prediction = lstm_q_model(inputs, training=False)
                q_table_for_step_t = q_tables_prediction[:, t, :]
                q_table_for_step_t = q_table_for_step_t.numpy()
                predicted_a_t = np.argmax(q_table_for_step_t, axis=1)
                actions[:, t] = predicted_a_t

                routes_selected = self.convert_actions_to_routes(path_indices=path_indices,
                                                                 block_id=t,
                                                                 actions_arr=predicted_a_t)
                path_selections = np.concatenate([path_selections[:, :-1],
                                                  routes_selected,
                                                  path_selections[:, -1][:, np.newaxis]], axis=1)
                path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)

            validity_vector = self.multiPathInfoObject.past_decisions_validity_array[path_indices]
            mac_vector = self.multiPathInfoObject.past_decisions_mac_array[path_indices[:-1]]
            validity_vectors.append(validity_vector)
            mac_vectors.append(mac_vector)

        validity_vector_complete = np.concatenate(validity_vectors)
        mac_vector_complete = np.concatenate(mac_vectors)
        accuracy = np.mean(validity_vector_complete)
        mac = np.mean(mac_vector_complete)
        print("Accuracy:{0} Mac:{1}".format(accuracy, mac))







