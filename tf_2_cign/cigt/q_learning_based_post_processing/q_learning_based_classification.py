import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.data_classes.multipath_routing_info2 import MultipathCombinationInfo2
from tf_2_cign.cigt.q_learning_based_post_processing.lstm_based_q_model import LstmBasedQModel
from tf_2_cign.cigt.q_learning_based_post_processing.q_learning_based_post_processing import QLearningRoutingOptimizer
from tf_2_cign.utilities.utilities import Utilities


class QLearningBasedRoutingClassification(QLearningRoutingOptimizer):
    intermediate_outputs_path = os.path.join(os.path.dirname(__file__), "..", "intermediate_outputs")

    def __init__(self, run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                 max_test_val_diff, random_seed):

        super().__init__(run_id, num_of_epochs, accuracy_weight, mac_weight, model_loader, model_id, val_ratio,
                         max_test_val_diff, random_seed)
        self.bceLosses = [tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.5)
                          for _ in range(len(self.pathCounts) - 1)]
        self.kind = "classification"

    def get_inputs_outputs_from_rnn(self, sample_indices, actions):
        inputs, q_tables_ground_truth = super(QLearningBasedRoutingClassification, self).get_inputs_outputs_from_rnn(
            sample_indices, actions
        )
        # Convert Q-table ground truths to classification labels
        labels_arr = []
        for idx in range(q_tables_ground_truth.shape[1]):
            q_table = q_tables_ground_truth[:, idx, :]
            lbl = np.argmax(q_table, axis=1)
            labels_arr.append(lbl)
        labels = tf.stack(labels_arr, axis=1)
        return inputs, labels

    def calculate_loss(self, lstm_q_model, inputs, q_tables_ground_truth):
        q_tables_prediction = lstm_q_model(inputs, training=True)
        losses_per_layer = []
        for t, q_predicted in enumerate(q_tables_prediction):
            q_truth = q_tables_ground_truth[:, t]
            bce_loss = self.bceLosses[t](q_truth, q_predicted[:, 0])
            print("Layer {0} BCE Loss:{1}".format(t, bce_loss.numpy()))
            losses_per_layer.append(bce_loss)
        total_loss = tf.reduce_mean(losses_per_layer)
        print("Total BCE Loss:{0}".format(total_loss))
        return total_loss

    def predict_actions(self, model, inputs, time_step):
        q_tables_prediction = model(inputs, training=False)
        q_table_for_step_t = q_tables_prediction[time_step]
        q_table_for_step_t = q_table_for_step_t.numpy()[:, 0]
        predicted_a_t = (q_table_for_step_t >= 0.5).astype(np.int32)
        return predicted_a_t

    def train_with_resampling_techniques(self):
        datasets = {}
        batch_size = 200
        step_count = len(self.model.pathCounts) - 1
        for dataset_type, indices in [("validation", self.valIndices), ("test", self.testIndices)]:
            dataset_tf = tf.data.Dataset.from_tensor_slices((indices,)).shuffle(1000).batch(batch_size)
            datasets[dataset_type] = dataset_tf

        X_dict = {}
        y_dict = {}
        for dataset_type in ["validation", "test"]:
            X_dict[dataset_type] = []
            y_dict[dataset_type] = []
            for tpl in datasets[dataset_type]:
                # print("iteration_num:{0}".format(iteration_num))
                sample_idx = tpl[0].numpy()
                # path_selections = sample_idx[:, np.newaxis]
                # q_table_selections = sample_idx[:, np.newaxis]
                # Step 1: Sample trajectories
                actions = np.random.randint(low=0, high=2, size=(sample_idx.shape[0], step_count))

                # Step 2: Create network inputs and outputs; raw routing features and optimal q_tables
                inputs, q_tables_ground_truth = self.get_inputs_outputs_from_rnn(sample_indices=sample_idx,
                                                                                 actions=actions)
                X_dict[dataset_type].append(inputs[0])
                y_dict[dataset_type].append(q_tables_ground_truth[:, 0].numpy())
            X_dict[dataset_type] = np.concatenate(X_dict[dataset_type], axis=0)
            y_dict[dataset_type] = np.concatenate(y_dict[dataset_type], axis=0)
            counter = Counter(y_dict[dataset_type])
            print("Dataset {0} distribution: {1}".format(dataset_type, counter))

        param_grid = \
            {
                "pca__n_components": [2, 8, 32, 128],
                "mlp__hidden_layer_sizes": [(32, ), (128, ), (16, 8), (32, 16, 8), (512, 256, 128)],
                "mlp__activation": ["relu"],
                "mlp__solver": ["adam"],
                # "mlp__learning_rate": ["adaptive"],
                "mlp__alpha": [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                "mlp__max_iter": [10000],
                "mlp__early_stopping": [True],
                "mlp__n_iter_no_change": [100]
                # "pca__n_components": [None],
                # "mlp__hidden_layer_sizes": [(512, 256, 128)],
                # "mlp__activation": ["relu"],
                # "mlp__solver": ["adam"],
                # # "mlp__learning_rate": ["adaptive"],
                # "mlp__alpha": [0.0001],
                # "mlp__max_iter": [10000],
                # "mlp__early_stopping": [True],
                # "mlp__n_iter_no_change": [100]
            }

        # search = GridSearchCV(pipe, param_grid, n_jobs=6, cv=10, verbose=10,
        #                       scoring=["accuracy", "f1_weighted", "f1_micro", "f1_macro",
        #                                "balanced_accuracy"],
        #                       refit="accuracy")
        smote_enn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smote_enn.fit_resample(X_dict["validation"], y_dict["validation"])
        counter2 = Counter(y_resampled)
        print(counter2)

        # "pca__n_components": [2, 8, 32, 128],
        # "mlp__hidden_layer_sizes": [(32,), (128,), (16, 8), (32, 16, 8), (512, 256, 128)],
        # "mlp__activation": ["relu"],
        # "mlp__solver": ["adam"],
        # "mlp__alpha": [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        # "mlp__max_iter": [10000],
        # "mlp__early_stopping": [True],
        # "mlp__n_iter_no_change": [100]

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

        score_list = []
        for tpl in param_grid:
            param_dict = {
                "pca__n_components": tpl[0],
                "mlp__hidden_layer_sizes": tpl[1],
                "mlp__activation": tpl[2],
                "mlp__solver": tpl[3],
                "mlp__alpha": tpl[4],
                "mlp__max_iter": tpl[5],
                "mlp__early_stopping": tpl[6],
                "mlp__n_iter_no_change": tpl[7]
            }
            standard_scaler = StandardScaler()
            pca = PCA()
            mlp = MLPClassifier(verbose=True)
            pipe = Pipeline(steps=[("scaler", standard_scaler),
                                   ('pca', pca),
                                   ('mlp', mlp)])
            pipe.set_params(**param_dict)
            pipe.fit(X_resampled, y_resampled)
            y_pred = {"resampled_validation": pipe.predict(X_resampled),
                      "validation": pipe.predict(X_dict["validation"]),
                      "test": pipe.predict(X_dict["test"])}

            print("*************Resampled Training*************")
            print(classification_report(y_pred=y_pred["resampled_validation"], y_true=y_resampled))
            print("*************Training*************")
            print(classification_report(y_pred=y_pred["validation"], y_true=y_dict["validation"]))
            print("*************Test*************")
            print(classification_report(y_pred=y_pred["test"], y_true=y_dict["test"]))
            macro_f1 = f1_score(y_true=y_dict["test"], y_pred=y_pred["test"], average='macro')
            micro_f1 = f1_score(y_true=y_dict["test"], y_pred=y_pred["test"], average='micro')
            weighted_f1 = f1_score(y_true=y_dict["test"], y_pred=y_pred["test"], average='weighted')
            score_list.append((macro_f1, param_dict))

        # Sort according to scores
        sorted_score_list = sorted(score_list, key=lambda t_: t_[0], reverse=True)
        print("X")






        # search.fit(X_resampled, y_resampled)
        # best_model = search.best_estimator_
        # best_params = search.best_params_
        # print(best_params)
        # # pipe.set_params(**param_grid)
        #
        # # best_model.fit(X_resampled, y_resampled)
        #
        # y_pred = {"resampled_validation": best_model.predict(X_resampled),
        #           "validation": best_model.predict(X_dict["validation"]),
        #           "test": best_model.predict(X_dict["test"])}
        #
        # print("*************Resampled Training*************")
        # print(classification_report(y_pred=y_pred["resampled_validation"], y_true=y_resampled))
        # print("*************Training*************")
        # print(classification_report(y_pred=y_pred["validation"], y_true=y_dict["validation"]))
        # print("*************Test*************")
        # print(classification_report(y_pred=y_pred["test"], y_true=y_dict["test"]))
        # print("X")
        #
        # # Step 3: Transform h outputs from each block into Q table outputs.
        # # with tf.GradientTape() as tape:
        # #     loss = self.calculate_loss(lstm_q_model=lstm_q_model, inputs=inputs,
        # #                                q_tables_ground_truth=q_tables_ground_truth)
        # #     # mse_loss = tf.with.mean_squared_error(q_tables_ground_truth, q_tables_prediction)
        # # grads = tape.gradient(loss, lstm_q_model.trainable_variables)
        # # self.optimizer.apply_gradients(zip(grads, lstm_q_model.trainable_variables))
        # # print("Lr:{0}".format(self.optimizer._decayed_lr(tf.float32).numpy()))
        # # iteration_num += 1
