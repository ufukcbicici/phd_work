import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_handling.usps_dataset import UspsDataset

dataset = UspsDataset(validation_sample_count=0)

X_train, y_train = dataset.trainingSamples, dataset.trainingLabels
X_test, y_test = dataset.testSamples, dataset.testLabels

mlp = MLPClassifier()
pipe = Pipeline(steps=[('mlp', mlp)])
param_grid = \
    [{
        "mlp__hidden_layer_sizes": [(64, 32, 16)],
        "mlp__activation": ["relu"],
        "mlp__solver": ["adam"],
        "mlp__batch_size": [125],
        # "mlp__learning_rate": ["adaptive"],
        "mlp__alpha": [0.0], #, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0],
        "mlp__max_iter": [10000],
        "mlp__early_stopping": [True],
        "mlp__n_iter_no_change": [100]
    }]
search = GridSearchCV(pipe, param_grid, n_jobs=8, cv=10, verbose=10,
                      scoring=["accuracy", "f1_weighted", "f1_micro", "f1_macro",
                               "balanced_accuracy"],
                      refit="accuracy")
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
model = search.best_estimator_
# Classify training and test sets
print("*************Training*************")
y_train_hat = model.predict(X_train)
print(classification_report(y_pred=y_train_hat, y_true=y_train))
f1_macro_train = f1_score(y_true=y_train, y_pred=y_train_hat, average="macro")
f1_micro_train = f1_score(y_true=y_train, y_pred=y_train_hat, average="micro")
print("*************Test*************")
y_test_hat = model.predict(X_test)
print(classification_report(y_pred=y_test_hat, y_true=y_test))
f1_macro_test = f1_score(y_true=y_test, y_pred=y_test_hat, average="macro")
f1_micro_test = f1_score(y_true=y_test, y_pred=y_test_hat, average="micro")
print("X")
