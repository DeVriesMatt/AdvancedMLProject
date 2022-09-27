import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  #plot first for training heartbeats. Each heartbeat is 188 long.

import os
for dirname, _, filenames in os.walk('/heartbeat'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/Users/mattdevries/PycharmProjects/AdvancedMLProject/heartbeat/mitbih_train.csv', header=None)
test = pd.read_csv('/Users/mattdevries/PycharmProjects/AdvancedMLProject/heartbeat/mitbih_test.csv', header=None)

print(train.shape)
print(test.shape)


plt.plot(train.iloc[0, :186])
plt.show()
plt.savefig("heartbeatplot.png")
plt.plot(train.iloc[1, :186])
plt.show()
plt.plot(train.iloc[2, :186])
plt.show()
plt.plot(train.iloc[3, :186])
plt.show()

print(train[187][0], train[187][1], train[187][2], train[187][3])
