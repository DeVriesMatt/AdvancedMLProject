import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


# ===========================================

models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'MLPClassifier': MLPClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

params = {
    'KNeighborsClassifier': {'n_neighbors': [1, 3, 5, 10, 15, 20]},
    'MLPClassifier': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'max_iter': [1000]},
    'AdaBoostClassifier': {'n_estimators': [20, 35, 50, 65, 80], 'learning_rate': [0.8, 1.0]},
}

helper1 = EstimatorSelectionHelper(models, params)
helper1.fit(XDATA, YDATA, scoring='f1', n_jobs=2)
helper1.score_summary(sort_by='max_score')

# ===========================================

heart_path = "./datasets/heartbeat/"
matlab_train = "./datasets/datasets_without_augmentation/matlab/train/"
matlab_test = "./datasets/datasets_without_augmentation/matlab/test/"
oversampled = "./datasets/oversampled/"
results = "./results/"

train_dir = []
test_dir = []
for train in sorted(os.listdir(oversampled)):
    train_dir.append(train)

for test in sorted(os.listdir(matlab_test)):
    test_dir.append(test)

print(train_dir)
print(test_dir)

# for dataset in range(3, len(test_dir)):
#
#     train = pd.read_csv(oversampled + train_dir[dataset], header=None)
#     test = pd.read_csv(matlab_test + test_dir[dataset], header=None)
#
#     train = train.iloc[1:, 1:].astype(float)
#     test = test.iloc[1:].astype(float)
#
#     X_train = train.iloc[:, 0:(train.shape[1]-1)]
#     y_train = train.iloc[:, train.shape[1]-1]
#
#     X_test = test.iloc[:, 0:(test.shape[1]-1)]
#     y_test = test.iloc[:, test.shape[1]-1]
#
#     print('\n--- Running classifiers for ---' + train_dir[dataset])
#     for name, clf in zip(names, classifiers):
#         print(name)
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         class_report = classification_report(y_test, y_pred, output_dict=True)
#         conf_mat_test = confusion_matrix(y_test, y_pred)
#         report = pd.DataFrame(class_report).transpose()
#         report.to_csv(results + 'report_' + name + "_" + train_dir[dataset])
#         cmat = pd.DataFrame(conf_mat_test)
#         cmat.to_csv(results + 'confus_' + name + "_" + train_dir[dataset])


for dataset in range(len(test_dir), len(train_dir)):

    train = pd.read_csv(oversampled + train_dir[dataset], header=None)
    test = pd.read_csv(matlab_test + test_dir[dataset - 7], header=None)

    train = train.iloc[1:, 1:].astype(float)
    test = test.iloc[1:].astype(float)

    X_train = train.iloc[:, 0:(train.shape[1] - 1)]
    y_train = train.iloc[:, train.shape[1] - 1]

    X_test = test.iloc[:, 0:(test.shape[1] - 1)]
    y_test = test.iloc[:, test.shape[1] - 1]

    print('\n--- Running classifiers for ---' + train_dir[dataset])
    for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        print(class_report)
        conf_mat_test = confusion_matrix(y_test, y_pred)
        report = pd.DataFrame(class_report).transpose()
        report.to_csv(results + 'report_' + name + "_" + train_dir[dataset])
        cmat = pd.DataFrame(conf_mat_test)
        cmat.to_csv(results + 'confus_' + name + "_" + train_dir[dataset])
