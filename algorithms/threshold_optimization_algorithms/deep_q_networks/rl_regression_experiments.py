import numpy as np
import pickle
from collections import Counter
import os

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

a_0_estimates = {}
Q_1 = {}
Q_1_hat = {}

for data_type in ["training", "test"]:
    f = open("rl_{0}_data.sav".format(data_type), "rb")
    data_dict = pickle.load(f)
    a_0_estimates[data_type] = data_dict["a_0_estimates"]
    Q_1[data_type] = data_dict["Q_1"]
    Q_1_hat[data_type] = data_dict["Q_1_hat"]
    f.close()

inf_marker = -1.0e10

action_counter = Counter(a_0_estimates["training"])
for action_id in action_counter.keys():
    Q_1_subset = Q_1["training"][a_0_estimates["training"] == action_id, :]
    Q_1_hat_subset = Q_1_hat["training"][a_0_estimates["training"] == action_id, :]
    non_zero_indices = np.nonzero(Q_1_subset != inf_marker)
    rows = non_zero_indices[0]
    cols = non_zero_indices[1]
    row_counter = Counter(rows)
    feature_dims = set(row_counter.values())
    assert len(feature_dims) == 1
    feature_dim = list(feature_dims)[0]
    X = Q_1_hat_subset[rows, cols]
    X = np.reshape(X, newshape=(X.shape[0] // feature_dim, feature_dim))
    Y = Q_1_subset[rows, cols]
    Y = np.reshape(Y, newshape=(Y.shape[0] // feature_dim, feature_dim))
    assert np.prod(Q_1_hat_subset[rows, cols].shape) == np.prod(X.shape)
    for feature_id in range(feature_dim):
        y_true = Y[:, feature_id]
        y_pred = X[:, feature_id]
        mse_prior = mean_squared_error(y_true, y_pred)
        print("mse_prior={0}".format(mse_prior))
        print("X")

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge())
        ])

        param_grid = \
            [{
                "ridge__alpha": [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100.0],
            }]

        grid_search = GridSearchCV(pipeline, param_grid, n_jobs=8, cv=10, verbose=10,
                                   scoring=["neg_mean_squared_error", "r2"], refit="neg_mean_squared_error")
        grid_search.fit(X, y_true)
        y_predicted_training = grid_search.best_estimator_.predict(X)
        mse_posterior = mean_squared_error(y_true, y_predicted_training)
        print("mse_posterior={0}".format(mse_posterior))
        print("X")



    # index_array = np.stack([rows, cols], axis=1)
    # rows_ordered = sorted(list(set(rows)))
    # for row_id in rows_ordered:
    #     row_entries = index_array[index_array[:, 0] == row_id, :]

print("X")
