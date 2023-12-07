import numpy as np
from tensorflow.python.keras import layers, models
import pandas as pd
import random
from dataUSE import dataDirect0
from misspredict import Improvedlstm
import matplotlib.pyplot as plt
from random import uniform


def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

dataframe = dataDirect0
dataframe1 = dataDirect0

dataTarget = []
for index, row in dataframe1.iterrows():
        time = row[0]
        dataTarget.append(time)
        '\n'
pd.set_option('display.max_rows', None)
# print(dataTarget)

rate = 0.2
numMiss = len(dataframe)*rate
# print(numMiss)
for i in range(0, int(numMiss)):
    colume = random.randint(0, len(dataframe))
    dataframe.loc[colume,'numDirect0'] = 'x'
# print(dataframe)

Datadir0 = []
for index, row in dataframe.iterrows():
        oneData = row[0]
        Datadir0.append(oneData)
        '\n'
pd.set_option('display.max_rows', None)
# print(Datadir0)

# fix random seed for reproducibility
np.random.seed(7)

# load data
dataset = Datadir0
# print(dataset)

# load target
targetset = dataTarget

# split into train and test sets
train_size = int(len(dataset) * 0.75)
train = dataset[0:train_size]
test = dataset[train_size:]
# print(train)
# print(test)

# target splitting
train_size1 = int(len(dataset) * 0.75)
train1 = targetset[0:train_size1]
test1 = targetset[train_size:]

Comdata = []
Missdata = []
for i in range(0, train_size):
    if train[i] == 'x':
        Comdata.append(0)
    else:
        Comdata.append(train[i])

for i in range(0, train_size):
    if train[i] == 'x':
        a = random.randrange(0, 20)
        Missdata.append(a)
    else:
        Missdata.append(0)
x = list_split(Comdata, 33)
sigema = list_split(Missdata, 33)
# print(sigema)

x = np.array(x)
y = np.array(x)
sigema = np.array(sigema)
x_train = np.reshape(x, (x.shape[0], x.shape[1], 1))
y_train = np.reshape(y, (y.shape[0], y.shape[1], 1))
sigema = np.reshape(sigema, (sigema.shape[0], sigema.shape[1], 1))

char_to_idx = x_train
idx_to_char = y_train

lstm = Improvedlstm(sigema, char_to_idx, idx_to_char)
loss, parame = lstm.optimize(sigema, char_to_idx, idx_to_char)
loss = loss
print(loss)
# print(parame)
fig1 = plt.figure(1)
plt.title('the change curve of loss 200 epochs')
plt.xlabel("epochs")
plt.ylabel("loss")
for i in range(0,1,199):
    plt.plot(loss, 'r')
    plt.show()