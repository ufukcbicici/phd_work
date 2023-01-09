import os
from collections import Counter
from multiprocessing import Process

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.data_classes.multipath_routing_info2 import MultipathCombinationInfo2
from tf_2_cign.cigt.q_learning_based_post_processing.lstm_based_q_model import LstmBasedQModel
from tf_2_cign.cigt.q_learning_based_post_processing.q_learning_based_classification import \
    QLearningBasedRoutingClassification
from tf_2_cign.cigt.q_learning_based_post_processing.q_learning_based_post_processing import QLearningRoutingOptimizer
from tf_2_cign.utilities.utilities import Utilities


class QLearningBasedCombinedClassification(QLearningBasedRoutingClassification):
    intermediate_outputs_path = os.path.join(os.path.dirname(__file__), "..", "intermediate_outputs")

    def __init__(self, run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                 max_test_val_diff, random_seed):

        super().__init__(run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                         max_test_val_diff, random_seed)
        self.kind = "classification"

    def get_inputs_outputs_from_rnn(self, sample_indices, actions):
        inputs, q_tables_ground_truth = \
            super(QLearningBasedCombinedClassification, self).get_inputs_outputs_from_rnn(
                sample_indices=sample_indices, actions=actions)
        q_tables_np = q_tables_ground_truth.cpu().numpy()
        q_tables_ground_truth_np = [q_tables_np[:, idx] for idx in range(q_tables_np.shape[1])]
        return inputs, q_tables_ground_truth_np

    def prepare_data(self):
        datasets = {}
        batch_size = 200
        step_count = len(self.model.pathCounts) - 1
        for dataset_type, indices in [("validation", self.valIndices), ("test", self.testIndices)]:
            dataset_tf = tf.data.Dataset.from_tensor_slices((indices,)).shuffle(1000).batch(batch_size)
            datasets[dataset_type] = dataset_tf

        # Combine data features for every possible trajectory. Train a single classifier for each one of them.
        choice_count = len(self.pathCounts) - 1
        choice_combinations = Utilities.get_cartesian_product(
            list_of_lists=[[0, 1] for _ in range(choice_count)])

        choice_grid = Utilities.get_cartesian_product(list_of_lists=[["validation", "test"],
                                                                     choice_combinations,
                                                                     [t for t in range(step_count)]])
        X_dict = {}
        y_dict = {}
        for tpl in choice_grid:
            X_dict[tpl] = []
            y_dict[tpl] = []
        print("X")

        for dataset_type in ["validation", "test"]:
            for tpl in datasets[dataset_type]:
                for choice_combination in choice_combinations:
                    sample_idx = tpl[0].numpy()
                    actions = np.expand_dims(np.array(choice_combination), axis=0)
                    actions = np.repeat(actions, axis=0, repeats=sample_idx.shape[0])
                    inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=sample_idx,
                                                                                     actions=actions)
                    for t in range(choice_count):
                        # X_dict[(dataset_type, choice_combination, t)]
                        X_dict[(dataset_type, choice_combination, t)].append(inputs[t])
                        y_dict[(dataset_type, choice_combination, t)].append(q_tables_ground_truth[t])

        for tpl in choice_grid:
            X_dict[tpl] = np.concatenate(X_dict[tpl], axis=0)
            y_dict[tpl] = np.concatenate(y_dict[tpl], axis=0)

        # Check correctness of the generated vectors
        past_combinations = set()
        for t in range(step_count):
            for comb in choice_combinations:
                past_combinations.add(comb[:t])

        final_datasets = {}
        for dataset_type in ["validation", "test"]:
            for past_combination in past_combinations:
                X_matching_arrays = []
                y_matching_arrays = []
                time_step = len(past_combination)
                for tpl in choice_grid:
                    if tpl[0] == dataset_type and tpl[1][:time_step] == past_combination and tpl[2] == time_step:
                        X_matching_arrays.append(X_dict[tpl])
                        y_matching_arrays.append(y_dict[tpl])
                X_mean = np.mean(np.stack(X_matching_arrays, axis=-1), axis=-1)
                for X_ in X_matching_arrays:
                    assert np.allclose(X_, X_mean)
                for y_ in y_matching_arrays:
                    assert np.array_equal(y_, y_matching_arrays[0])
                final_datasets[(dataset_type, past_combination)] = (X_mean, y_matching_arrays[0])
        return datasets, final_datasets

    # @staticmethod
    # def evaluate_routing_performance(y, y_pred, **kwargs):
    #     step_count = len(self.model.pathCounts) - 1
    #     validity_vectors = []
    #     mac_vectors = []
    #     for tpl in dataset:
    #         sample_indices = tpl[0].numpy()
    #         actions = np.zeros(shape=(sample_indices.shape[0], step_count), dtype=np.int32)
    #         path_selections = sample_indices[:, np.newaxis]
    #         path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
    #         # A very suboptimal algorithm:
    #         # Given a batch of samples, we load an actions array of the size BxS, where B is the batch size,
    #         # S is the step count, with zeros.
    #         # At the start, where t=0, we use the initial actions array.
    #         # The q_table output at t=0, will act as our guidance for the step t=1. We pick a_0 = argmax(q_0, axis=1).
    #         # Then load actions[:, 0] <- a_0. This means we are going to select s_1 according to the actions we have
    #         # just sampled. Then we run our inference again. The q_table output at t=1 will now show q_1, with respect
    #         # to actions at a_0. Then we pick a_1 = argmax(q_1, axis=1).
    #         # Without loss of generality, if we assume that we have 2 time steps, with a_1 is determined, we can pick
    #         # the optimal selected path from the start to end and then calculate the accuracy and MAC burden.
    #
    #         for t in range(step_count):
    #             inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=sample_indices,
    #                                                                              actions=actions)
    #             predicted_a_t = self.predict_actions(lstm_q_model=model, inputs=inputs, time_step=t)
    #             actions[:, t] = predicted_a_t
    #
    #             routes_selected = self.convert_actions_to_routes(path_indices=path_indices,
    #                                                              block_id=t,
    #                                                              actions_arr=predicted_a_t)
    #             path_selections = np.concatenate([path_selections[:, :-1],
    #                                               routes_selected,
    #                                               path_selections[:, -1][:, np.newaxis]], axis=1)
    #             path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
    #
    #         validity_vector = self.multiPathInfoObject.past_decisions_validity_array[path_indices]
    #         mac_vector = self.multiPathInfoObject.past_decisions_mac_array[path_indices[:-1]]
    #         validity_vectors.append(validity_vector)
    #         mac_vectors.append(mac_vector)

    def predict_actions(self, model, inputs, time_step):
        X_t = inputs[time_step]
        predicted_a_t = model(X_t)
        # q_tables_prediction = model(inputs, training=False)
        # q_table_for_step_t = q_tables_prediction[time_step]
        # q_table_for_step_t = q_table_for_step_t.numpy()[:, 0]
        # predicted_a_t = (q_table_for_step_t >= 0.5).astype(np.int32)
        return predicted_a_t

    @staticmethod
    def evaluate_multipath_accuracy(model, dataset, multipath_object):
        step_count = len(multipath_object.pathCounts) - 1
        validity_vectors = []
        mac_vectors = []
        actions = []
        features = multipath_object.past_decisions_h_features_list
        for tpl in dataset:
            sample_indices = tpl[0].numpy()
            path_selections = sample_indices[:, np.newaxis]
            path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)

            for t in range(step_count):
                h_t = features[t][path_indices]
                a_t = model.predict(h_t)
                actions.append(a_t)
                # Convert actions to routing decisions
                routing_probabilities = multipath_object.past_decisions_routing_probabilities_list[t][path_indices]
                ig_routings = Utilities.one_hot_numpy(arr=routing_probabilities)
                all_routings = np.ones_like(ig_routings)
                routes_selected = np.where(a_t[:, np.newaxis], all_routings, ig_routings)
                path_selections = np.concatenate([path_selections[:, :-1],
                                                  routes_selected,
                                                  path_selections[:, -1][:, np.newaxis]], axis=1)
                path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
            validity_vector = multipath_object.past_decisions_validity_array[path_indices]
            mac_vector = multipath_object.past_decisions_mac_array[path_indices[:-1]]
            validity_vectors.append(validity_vector)
            mac_vectors.append(mac_vector)
        validity_vector_complete = np.concatenate(validity_vectors)
        mac_vector_complete = np.concatenate(mac_vectors)
        accuracy = np.mean(validity_vector_complete)
        mac = np.mean(mac_vector_complete)
        mac_relative = mac / np.nanmin(multipath_object.past_decisions_mac_array)
        print("Accuracy:{0} Mac:{1}".format(accuracy, mac_relative))
        return accuracy, mac_relative

        # # Select the possible routing options
        # # If block_choice = 0 -> ig selection
        # ig_routings = Utilities.one_hot_numpy(arr=routing_probabilities)
        # # If block_choice = 1 -> all paths
        # all_routings = np.ones_like(ig_routings)
        # routes_selected = np.where(actions_arr[:, np.newaxis], all_routings, ig_routings)

        #     inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=sample_indices,
        #                                                                      actions=actions)
        #     predicted_a_t = self.predict_actions(lstm_q_model=model, inputs=inputs, time_step=t)
        #     actions[:, t] = predicted_a_t
        #
        #     routes_selected = self.convert_actions_to_routes(path_indices=path_indices,
        #                                                      block_id=t,
        #                                                      actions_arr=predicted_a_t)
        #     path_selections = np.concatenate([path_selections[:, :-1],
        #                                       routes_selected,
        #                                       path_selections[:, -1][:, np.newaxis]], axis=1)
        #     path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
        #
        # validity_vector = self.multiPathInfoObject.past_decisions_validity_array[path_indices]
        # mac_vector = self.multiPathInfoObject.past_decisions_mac_array[path_indices[:-1]]
        # validity_vectors.append(validity_vector)
        # mac_vectors.append(mac_vector)

        # for tpl in dataset:
        #     sample_indices = tpl[0].numpy()
        #     actions = np.zeros(shape=(sample_indices.shape[0], step_count), dtype=np.int32)
        #     path_selections = sample_indices[:, np.newaxis]
        #     path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
        #     # A very suboptimal algorithm:
        #     # Given a batch of samples, we load an actions array of the size BxS, where B is the batch size,
        #     # S is the step count, with zeros.
        #     # At the start, where t=0, we use the initial actions array.
        #     # The q_table output at t=0, will act as our guidance for the step t=1. We pick a_0 = argmax(q_0, axis=1).
        #     # Then load actions[:, 0] <- a_0. This means we are going to select s_1 according to the actions we have
        #     # just sampled. Then we run our inference again. The q_table output at t=1 will now show q_1, with respect
        #     # to actions at a_0. Then we pick a_1 = argmax(q_1, axis=1).
        #     # Without loss of generality, if we assume that we have 2 time steps, with a_1 is determined, we can pick
        #     # the optimal selected path from the start to end and then calculate the accuracy and MAC burden.
        #
        #     for t in range(step_count):
        #         inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=sample_indices,
        #                                                                          actions=actions)
        #         predicted_a_t = self.predict_actions(lstm_q_model=model, inputs=inputs, time_step=t)
        #         actions[:, t] = predicted_a_t
        #
        #         routes_selected = self.convert_actions_to_routes(path_indices=path_indices,
        #                                                          block_id=t,
        #                                                          actions_arr=predicted_a_t)
        #         path_selections = np.concatenate([path_selections[:, :-1],
        #                                           routes_selected,
        #                                           path_selections[:, -1][:, np.newaxis]], axis=1)
        #         path_indices = Utilities.convert_trajectory_array_to_indices(trajectory_array=path_selections)
        #
        #     validity_vector = self.multiPathInfoObject.past_decisions_validity_array[path_indices]
        #     mac_vector = self.multiPathInfoObject.past_decisions_mac_array[path_indices[:-1]]
        #     validity_vectors.append(validity_vector)
        #     mac_vectors.append(mac_vector)
        #
        # validity_vector_complete = np.concatenate(validity_vectors)
        # mac_vector_complete = np.concatenate(mac_vectors)
        # accuracy = np.mean(validity_vector_complete)
        # mac = np.mean(mac_vector_complete)
        # print("Accuracy:{0} Mac:{1}".format(accuracy, mac))
        # return accuracy

    @staticmethod
    def train_with_params(process_id, X_val_resampled, y_val_resampled,
                          X_val, y_val, X_test, y_test, params_list, multipath_object, val_indices,
                          test_indices, batch_size, run_id, model_id):
        print("Process {0} has started.".format(process_id))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_indices,)).shuffle(1000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_indices,)).shuffle(1000).batch(batch_size)

        score_list = []
        for params in params_list:
            standard_scaler = StandardScaler()
            pca = PCA()
            mlp = MLPClassifier(verbose=True)
            pipe = Pipeline(steps=[("scaler", standard_scaler),
                                   ('pca', pca),
                                   ('mlp', mlp)])
            param_dict = {
                "pca__n_components": params[0],
                "mlp__hidden_layer_sizes": params[1],
                "mlp__activation": params[2],
                "mlp__solver": params[3],
                "mlp__alpha": params[4],
                "mlp__max_iter": params[5],
                "mlp__early_stopping": params[6],
                "mlp__n_iter_no_change": params[7]
            }
            pipe.set_params(**param_dict)
            pipe.fit(X_val_resampled, y_val_resampled)
            y_pred = {"resampled_validation": pipe.predict(X_val_resampled),
                      "validation": pipe.predict(X_val),
                      "test": pipe.predict(X_test)}
            print("*************Resampled Training*************")
            resampled_validation_report \
                = classification_report(y_pred=y_pred["resampled_validation"], y_true=y_val_resampled)
            print(resampled_validation_report)

            print("*************Training*************")
            validation_report = classification_report(y_pred=y_pred["validation"], y_true=y_val)
            print(validation_report)

            print("*************Test*************")
            test_report = classification_report(y_pred=y_pred["test"], y_true=y_test)
            print(test_report)

            macro_f1 = f1_score(y_true=y_test, y_pred=y_pred["test"], average='macro')
            micro_f1 = f1_score(y_true=y_test, y_pred=y_pred["test"], average='micro')
            weighted_f1 = f1_score(y_true=y_test, y_pred=y_pred["test"], average='weighted')
            score_list.append((macro_f1, param_dict))
            val_accuracy, val_mac = QLearningBasedCombinedClassification.evaluate_multipath_accuracy(
                model=pipe, dataset=val_dataset, multipath_object=multipath_object)
            test_accuracy, test_mac = QLearningBasedCombinedClassification.evaluate_multipath_accuracy(
                model=pipe, dataset=test_dataset, multipath_object=multipath_object)

            params_kv_list = sorted([(k, v) for k, v in param_dict.items()], key=lambda tpl: tpl[0])
            params_string = ""
            for tpl in params_kv_list:
                params_string += "{0}:{1}\n".format(tpl[0], tpl[1])

            DbLogger.write_into_table(table="cigt_q_learning", rows=[(
                run_id,
                model_id,
                val_accuracy,
                test_accuracy,
                val_mac,
                test_mac,
                params_string,
                macro_f1,
                resampled_validation_report,
                validation_report,
                test_report)])

    def train_with_resampling_techniques(self):
        datasets, final_datasets = self.prepare_data()
        past_combinations = list(set([tpl[1] for tpl in final_datasets.keys()]))

        X_val = []
        y_val = []
        X_test = []
        y_test = []
        for past_combination in past_combinations:
            X_val.append(final_datasets[("validation", past_combination)][0])
            y_val.append(final_datasets[("validation", past_combination)][1])
            X_test.append(final_datasets[("test", past_combination)][0])
            y_test.append(final_datasets[("test", past_combination)][1])
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        # Apply resampling
        smote_enn = SMOTEENN(random_state=0)
        X_val_resampled, y_val_resampled = smote_enn.fit_resample(X_val, y_val)
        counter2 = Counter(y_val_resampled)
        print(counter2)

        param_grid = \
            {
                "pca__n_components": [2, 8, 32, 64, 128],
                "mlp__hidden_layer_sizes": [(16,), (32,), (128,), (16, 8), (32, 16, 8), (512, 256, 128)],
                "mlp__activation": ["relu"],
                "mlp__solver": ["adam"],
                # "mlp__learning_rate": ["adaptive"],
                "mlp__alpha": [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                "mlp__max_iter": [10000],
                "mlp__early_stopping": [True],
                "mlp__n_iter_no_change": [100]
            }

        param_grid = Utilities.get_cartesian_product(list_of_lists=[
            param_grid["pca__n_components"],
            param_grid["mlp__hidden_layer_sizes"],
            param_grid["mlp__activation"],
            param_grid["mlp__solver"],
            param_grid["mlp__alpha"],
            param_grid["mlp__max_iter"],
            param_grid["mlp__early_stopping"],
            param_grid["mlp__n_iter_no_change"]
        ])

        number_of_jobs = 1
        chunks = Utilities.divide_array_into_chunks(param_grid, count=number_of_jobs)
        list_of_processes = []
        for process_id in range(number_of_jobs):
            process = Process(target=QLearningBasedCombinedClassification.train_with_params,
                              args=(process_id, X_val_resampled, y_val_resampled,
                                    X_val, y_val, X_test, y_test, chunks[process_id],
                                    self.multiPathInfoObject, self.valIndices, self.testIndices, 200, self.runId,
                                    self.modelId))
            list_of_processes.append(process)
            process.start()

        for process in list_of_processes:
            process.join()
