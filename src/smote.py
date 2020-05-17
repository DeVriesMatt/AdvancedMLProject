from imblearn.over_sampling import SMOTE
import os
import pandas as pd
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


# List the datasets contained in heartbeats
print(os.listdir("heartbeat"))

# Import the train and test datsets
train = pd.read_csv("heartbeat/mitbih_train.csv", header=None)
test = pd.read_csv("heartbeat/mitbih_test.csv", header=None)
df = pd.concat([train, test], axis=0)

# Print the number of observations of each class
train[187] = train[187].astype(int)
equilibre = train[187].value_counts()
print("The value counts for the training set before SMOTE:")
print(equilibre)


# Setting up the variables
target_train = train[187]
target_test = test[187]
y_train = to_categorical(target_train)
y_test = to_categorical(target_test)
print("Y train shape: {}".format(y_train.shape))

X_train = train.iloc[:, :186].values
X_test = test.iloc[:, :186].values
print("X train shape: {}".format(X_train.shape))

# Perform SMOTE firstly on only the training set
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

y_res_pd = pd.DataFrame(y_res)
print("The value counts for the training set after SMOTE:")
print(y_res_pd.sum(axis=0))

print(y_res.shape)

# Now perform smote on the whole dataset as done in https://link.springer.com/content/pdf/10.1007/s13246-019-00815-9.pdf
df[187] = df[187].astype(int)
equilibre = df[187].value_counts()
print(equilibre)

target = df[187]
y = to_categorical(target)
print("Y shape of whole dataset: {}".format(y.shape))

X = df.iloc[:, :186].values
print("X shape of whole dataset: {}".format(X.shape))

# Perform SMOTE firstly on only the training set
sm = SMOTE(random_state=42)
X_res_whole, y_res_whole = sm.fit_resample(X, y)

y_res_whole_pd = pd.DataFrame(y_res_whole)
print("The value counts for the training set after SMOTE:")
print(y_res_whole_pd.sum(axis=0))


#
# plt.figure(figsize=(20,10))
# my_circle=plt.Circle( (0,0), 0.7, color='white')
# plt.pie(equilibre, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.show()
