from imblearn.over_sampling import SMOTE, ADASYN
import os
import pandas as pd
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

heart_path = "./datasets/heartbeat/"
matlab_path = "./datasets/datasets_without_augmentation/matlab/train/"
oversampled = "./datasets/oversampled/"

# List the datasets contained in heartbeats
print(os.listdir(heart_path))

# Import the train and test datsets
train = pd.read_csv(matlab_path + "mitbih_train_without4.csv", header=None)
train = train.iloc[1:]

# Print the number of observations of each class
train[187] = train[187].astype(int)
equilibre = train[187].value_counts()
print("The value counts for the training set before SMOTE:")
print(equilibre)

plt.figure(figsize=(20, 10))
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(equilibre, labels=['N', 'V', 'S', 'F'],
        colors=['red', 'green', 'blue', 'skyblue'],
        autopct='%1.1f%%')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# Oversample the training sets in matlab folder
for filename in os.listdir(matlab_path):
        # Import the train and test datsets
        train = pd.read_csv(matlab_path + filename, header=None)

        train = train.iloc[1:]
        print(train.shape)

        # Print the number of observations of each class
        train[(train.shape[1]-1)] = train[(train.shape[1]-1)].astype(int)
        equilibre = train[(train.shape[1]-1)].value_counts()
        print("The value counts for the training set before SMOTE:")
        print(equilibre)

        # Setting up the variables
        target_train = train[(train.shape[1]-1)]
        y_train = to_categorical(target_train)
        print("Y train shape: {}".format(y_train.shape))

        X_train = train.iloc[:, :(train.shape[1]-2)].values
        print("X train shape: {}".format(X_train.shape))

        # Perform SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        X_res_pd = pd.DataFrame(X_res)
        y_res_pd = pd.DataFrame(y_res)
        print("The value counts for the training set after SMOTE:")
        print(y_res_pd.sum(axis=0))
        df = pd.concat([X_res_pd, y_res_pd], axis=1)
        df.to_csv(oversampled + "smote_" + filename)

        # Perform Adasyn
        ada = ADASYN(random_state=42)
        X_res_ada, y_res_ada = ada.fit_resample(X_train, y_train)

        X_res_pd_ada = pd.DataFrame(X_res_ada)
        y_res_pd_ada = pd.DataFrame(y_res_ada)
        print("The value counts for the whole dataset after ADASYN:")
        print(y_res_pd_ada.sum(axis=0))
        df = pd.concat([X_res_pd, y_res_pd], axis=1)
        df.to_csv(oversampled + "adasyn_" + filename)

