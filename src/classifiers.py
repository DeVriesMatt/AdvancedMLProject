from time import sleep

import pandas as pd
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = ["Nearest Neighbors",
         "Linear SVM",
         # "RBF SVM",
         # "Gaussian Process",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes"]

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

datapath = "./datasets/datasets_without_augmentation/matlab"
train = pd.read_csv(datapath + "/train/mitbih_train_without4.csv", header=None)
test = pd.read_csv(datapath + "/test/mitbih_test_without4.csv", header=None)

train = train.iloc[1:].astype(float)
test = test.iloc[1:].astype(float)
print(train.head())

print(train.shape)
X_train = train.iloc[:, 0:186]
y_train = train.iloc[:, 187]

X_test = test.iloc[:, 0:186]
y_test = test.iloc[:, 187]


def print_pause(text):
    sleep(0.5)
    print(text)
    sleep(0.5)


# Run classifiers
print_pause('\n--- Running classifiers ---')
methods = []
all_results = []


for name, clf in zip(names, classifiers):
    print_pause(name)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    r = [name]
    print(train_score)
    print(test_score)
    print(classification_report(y_test, y_pred))
    # all_results.append(r)

print(all_results)


