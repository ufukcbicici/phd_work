import numpy as np
import imblearn
from sklearn.base import BaseEstimator, RegressorMixin
from collections import Counter

from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import imblearn
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# "mlp__max_iter": [10000],
# "mlp__early_stopping": [True],
# "mlp__n_iter_no_change": [25]


class EnsembleResamplerRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 max_cluster_size,
                 max_upsample_ratio,
                 regressor_count,
                 regressor_shape,
                 alpha):
        self.max_cluster_size = max_cluster_size
        self.max_upsample_ratio = max_upsample_ratio
        self.regressor_count = regressor_count
        self.regressor_shape = regressor_shape
        self.alpha = alpha
        self.regressors = []

    def prepare_train_set(self, X, Q):
        vector_input = False
        if len(Q.shape) == 1:
            Q = Q[:, np.newaxis]
            vector_input = True
        q_counter = Counter([tuple(arr) for arr in Q])
        q_counter_numeric = {k.__hash__(): v for k, v in q_counter.items()}
        q_inverse_hash_map = {arr.__hash__(): arr for arr in q_counter.keys()}
        le = LabelEncoder()
        le.fit([tpl_code for tpl_code in q_counter_numeric.keys()])
        Q_labels = le.transform([tuple(arr).__hash__() for arr in Q])
        under_sample_counts = {
            le.transform([tpl_code])[0]:
                min(self.max_cluster_size, cnt) for tpl_code, cnt in q_counter_numeric.items()}
        under_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=under_sample_counts)
        X_hat, Q_labels_hat = under_sampler.fit_resample(X=X, y=Q_labels)
        # over_sample_counts = {
        #     tpl_code: min(int(self.max_upsample_ratio * cnt), self.max_cluster_size)
        #     for tpl_code, cnt in under_sample_counts.items()}
        # neighbor_count = 10
        # while neighbor_count > 0:
        #     print("Trying with neighbor count:{0}".format(neighbor_count))
        #     try:
        #         over_sampler = imblearn.over_sampling.SMOTE(sampling_strategy=over_sample_counts,
        #                                                     n_jobs=1, k_neighbors=neighbor_count)
        #         X_hat_2, Q_labels_hat_2 = over_sampler.fit_resample(X=X_hat, y=Q_labels_hat)
        #     except:
        #         print("Failed with neighbor count:{0}".format(neighbor_count))
        #         neighbor_count = neighbor_count-1
        #         continue
        #     break
        # print("------->Fitted with neighbor count:{0}".format(neighbor_count))
        hash_ids = le.inverse_transform(Q_labels_hat)
        Q_hat = np.stack([q_inverse_hash_map[hash_id] for hash_id in hash_ids], axis=0)
        if vector_input:
            Q_hat = Q_hat[:, 0]
        return X_hat, Q_hat

    def fit(self, X, y):
        # print("X shape={0}".format(X.shape))
        # print("y shape={0}".format(y.shape))
        for regressor_id in range(self.regressor_count):
            print("Fitting Regressor:{0}".format(regressor_id))
            X_hat, Q_hat = self.prepare_train_set(X=X, Q=y)
            mlp = MLPRegressor(
                hidden_layer_sizes=self.regressor_shape,
                activation="relu", solver="adam", alpha=self.alpha, max_iter=10000, early_stopping=True,
                n_iter_no_change=100, verbose=False)
            mlp.fit(X_hat, Q_hat)
            self.regressors.append(mlp)
        return self

    def predict(self, X):
        # print("X shape={0}".format(X.shape))
        y_predictions = []
        for regressor_id in range(self.regressor_count):
            y_id = self.regressors[regressor_id].predict(X)
            y_predictions.append(y_id)
        y_concat = np.stack(y_predictions, axis=-1)
        y_result = np.mean(y_concat, axis=-1)
        return y_result
