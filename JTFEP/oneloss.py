import numpy as np
import pandas as pd
from dataUSE import dataDirect0
from classmissran import Updatedata, Jointprediction, Samplelstm
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import random

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

dataframe = dataDirect0

rate = 0.2
numMiss = len(dataframe) * rate
for j in range(0, int(numMiss)):
    colume = random.randint(0, len(dataframe)-1)
    dataframe.loc[colume, 'numDirect0'] = np.nan
# print(dataframe)

Datadirtm = []
for index, row in dataframe.iterrows():
    oneData = row[0]
    Datadirtm.append(oneData)
    '\n'
pd.set_option('display.max_rows', None)
# print(Datadirt0)

# load data
dataset = Datadirtm
# print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.75)
train = dataset[0:train_size]
test = dataset[train_size:]
print(train)
# print(test)

Comdata = []
Missdata = []
for i in train:
    if np.isnan(i):
        Comdata.append(0)
    else:
        Comdata.append(i)
train1 = list_split(train, 33)
Comdata = list_split(Comdata, 33)
# print(Comdata)
imputer = KNNImputer(n_neighbors=5)
train = imputer.fit_transform(train1)
Missdata = train - Comdata
# print(Missdata)
traindata = np.array(train).flatten()
print(traindata.shape[0])

test1 = list_split(test, 33)
Cotest = []
Mitest = []
for i in test:
    if np.isnan(i):
        Cotest.append(0)
    else:
        Cotest.append(i)
Cotest = list_split(Cotest, 33)
imputer = KNNImputer(n_neighbors=11)
test = imputer.fit_transform(test1)
Mitest = test - Cotest
testdata = np.array(test).flatten()
Missdata = Updatedata(Comdata, Missdata, traindata)
r, l, t, pr = Jointprediction(Comdata, Missdata, traindata, Cotest, Mitest, testdata)
p, e, pe = Samplelstm(train, traindata, test, testdata)

fig1 = plt.figure()
plt.plot(r, 'red', label = 'JTFEP')
plt.plot(testdata, 'green', label = 'True Background Data')
plt.xlabel("Sampling time ID")
plt.ylabel("Traffic Flow")
# plt.title('MSE of the three methods under NMAR')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()
fig2 = plt.figure()
plt.plot(p, 'b', label = 'JTFEP')
plt.plot(testdata, 'green', label = 'True Background Data')
plt.xlabel("Sampling time ID")
plt.ylabel("Traffic Flow")
# plt.title('MSE of the three methods under NMAR')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()

print(l)
print(pe)