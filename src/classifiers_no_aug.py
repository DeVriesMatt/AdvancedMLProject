import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = ["NearestNeighbors",
         "LinearSVM",
         # "RBF SVM",
         # "Gaussian Process",
         "DecisionTree",
         "RandomForest",
         "NeuralNet",
         "AdaBoost",
         "NaiveBayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

heart_path = "./datasets/heartbeat/"
matlab_train = "./datasets/datasets_without_augmentation/matlab/train/"
matlab_test = "./datasets/datasets_without_augmentation/matlab/test/"
oversampled = "./datasets/oversampled/"
results = "./results/"


train_dir = []
test_dir = []
for train in sorted(os.listdir(matlab_train)):
    train_dir.append(train)

for test in sorted(os.listdir(matlab_test)):
    test_dir.append(test)

print(train_dir)
print(test_dir)


for dataset in range(len(train_dir)):

    train = pd.read_csv(matlab_train + train_dir[dataset], header=None)
    test = pd.read_csv(matlab_test + test_dir[dataset], header=None)

    train = train.iloc[1:].astype(float)
    test = test.iloc[1:].astype(float)

    X_train = train.iloc[:, 0:(train.shape[1]-1)]
    y_train = train.iloc[:, train.shape[1]-1]

    X_test = test.iloc[:, 0:(test.shape[1]-1)]
    y_test = test.iloc[:, test.shape[1]-1]

    print('\n--- Running classifiers for ---' + train_dir[dataset])
    for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(X_train, y_train)
        # train_score = clf.score(X_train, y_train)
        # test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_mat_test = confusion_matrix(y_test, y_pred)
        # print(train_score)
        # print(test_score)
        # print(class_report)
        report = pd.DataFrame(class_report).transpose()
        report.to_csv(results + 'report_' + name + "_" + test_dir[dataset])
        cmat = pd.DataFrame(conf_mat_test)
        cmat.to_csv(results + 'confus_' + name + "_" + test_dir[dataset])
