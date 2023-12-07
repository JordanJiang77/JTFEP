import tensorflow as tf
import numpy as np
import os

from Lstmcla import LSTM
from sklearn.impute import KNNImputer
import pandas as pd
import random
from dataUSE import dataDirect0
from misspredict import Improvedlstm
import matplotlib.pyplot as plt
from random import uniform
from dataUSE import Datadir0

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def relu(x):
    x = (np.abs(x) + x) / 2.0
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt, a_prev, c_prev):
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros((n_a + n_y, m))
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = np.multiply(ot, np.tanh(c_next))

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = relu(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

    return a_next, c_next, yt_pred, cache

def lstm_forward(Wy, x, a0):
    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache, = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt=x[:, :, t],
                                                        a_prev=a_next, c_prev=c_next)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:, :, t] = yt
        # Save the value of the next cell state (≈1 line)
        c[:, :, t] = c_next
        # Append the cache into caches (≈1 line)
        caches.append(cache)

        # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches

def compute_loss(y_hat, y):
    loss = 0
    a = len(y_hat)
    for i in range(a):
        loss += (y[i]-y_hat[i])**2
    return loss/a

losspredit = []
epoches = 10
# print(numMiss)

lossilstm = []
lossclstm = []

rate = 0.2
dataset = Datadir0

# split into train and test sets
train_size = int(len(dataset) * 0.75)
train = dataset[33:train_size]
test = dataset[train_size:]
# print(train)
# print(test)

train_size = train_size - 33
numMiss = train_size * rate
for j in range(int(numMiss)):
    m = random.randint(0, train_size - 1)
    train[m] = np.nan
# print(train)
train1 = train

train = list_split(train, 33)
imputer = KNNImputer(n_neighbors=5)
train = imputer.fit_transform(train)

Comdata = []
Missdata = []
for i in range(0, train_size):
    if train1[i] is np.nan:
        Comdata.append(0)
    else:
        Comdata.append(train1[i])

for i in range(0, train_size):
    if train1[i] is np.nan:
        a = random.randrange(0, 10)
        Missdata.append(a)
    else:
        Missdata.append(0)
x = list_split(Comdata, 33)
# sigema = list_split(Missdata, 33)
sigema = train - x
# print(x)

x = np.array(x)
y = np.array(x)
sigema = np.array(sigema)
x_train = np.reshape(x, (x.shape[1], 1, x.shape[0]))
y_train = np.reshape(y, (y.shape[1], 1, y.shape[0]))
sigema = np.reshape(sigema, (sigema.shape[1], 1, sigema.shape[0]))
lossi = []
char_to_idx = x_train
idx_to_char = y_train
lstm = Improvedlstm(sigema, char_to_idx, idx_to_char)
loss, parame = lstm.optimize(sigema, char_to_idx, idx_to_char)
# print(loss)
print(parame['sigema'])

Wy = parame["Wy"]
Wf = parame["Wf"]
Wi = parame["Wi"]
Wc = parame["Wc"]
Wo = parame["Wo"]
by = parame["by"]
bf = parame["bf"]
bi = parame["bi"]
bc = parame["bc"]
bo = parame["bo"]
dataset1 = Datadir0
# split into train and test sets
train_size = int(len(dataset1) * 0.75)
test1 = len(dataset1) - train_size
predictall = np.zeros(33, dtype=float)
testall = np.zeros(33, dtype=float)
loss0 = []
for b in range(0, test1 - 33, 33):
    test = dataset1[train_size + b:train_size + b + 33]
    # print(train)
    # print(test)

    t = list_split(test, 33)
    t = np.array(t)
    Testbatch = np.reshape(t, (t.shape[0], t.shape[1], 1))
    n_x, m, T_x = Testbatch.shape
    a_prev = dataset1[train_size + b - 33:train_size + b]
    a_prev = np.array(a_prev)
    c_prev = a_prev
    x_batch = Testbatch
    y_batch = Testbatch
    a, c, y_pred, caches = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, t, a_prev, c_prev)
    y_pred = y_pred[32]
    test = dataset1[train_size + b:train_size + b + 33]
    lossa = compute_loss(y_pred, test)
    loss0.append(lossa)
    print(lossa)
    # plot predicti
    fig = plt.figure()
    plt.plot(y_pred, 'r', label='Predicted flow with improved method missing rate 0.2')
    plt.plot(test, 'g', label='Real flow')
    plt.xlabel("Sensor ID")
    plt.ylabel("Traffic Flow (vehicles/min)")
    plt.title('Traffic data of different detectors at one time')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

train = list_split(train1, 33)
imputer = KNNImputer(n_neighbors=19)
train = imputer.fit_transform(train)
x = train
x = np.array(x)
y = np.array(x)

x_train = np.reshape(x, (x.shape[1], 1, x.shape[0]))
y_train = np.reshape(y, (y.shape[1], 1, y.shape[0]))
lossi = []
char_to_idx = x_train
idx_to_char = y_train
lstm = LSTM(char_to_idx, idx_to_char)
loss, parame = lstm.optimize(char_to_idx, idx_to_char)
# print(loss)

Wy = parame["Wy"]
Wf = parame["Wf"]
Wi = parame["Wi"]
Wc = parame["Wc"]
Wo = parame["Wo"]
by = parame["by"]
bf = parame["bf"]
bi = parame["bi"]
bc = parame["bc"]
bo = parame["bo"]
dataset1 = Datadir0
# split into train and test sets
train_size = int(len(dataset1) * 0.75)
test1 = len(dataset1) - train_size
predictall = np.zeros(33, dtype=float)
testall = np.zeros(33, dtype=float)
loss1 = []
for b in range(0, test1 - 33, 33):
    test = dataset1[train_size + b:train_size + b + 33]
    # print(train)
    # print(test)

    t = list_split(test, 33)
    t = np.array(t)
    Testbatch = np.reshape(t, (t.shape[0], t.shape[1], 1))
    n_x, m, T_x = Testbatch.shape
    a_prev = dataset1[train_size + b - 33:train_size + b]
    a_prev = np.array(a_prev)
    c_prev = a_prev
    x_batch = Testbatch
    y_batch = Testbatch
    a, c, y_pred, caches = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, t, a_prev, c_prev)
    y_pred = y_pred[32]
    test = dataset1[train_size + b:train_size + b + 33]
    lossa = compute_loss(y_pred, test)
    loss1.append(lossa)
    print(lossa)
    # plot predicti

ls0 = 0
for i in range(17):
    ls0 = ls0 + loss0[i]
ls0 = ls0 / 17
ls1 = 0
for i in range(17):
    ls1 = ls1 + loss1[i]
ls1 = ls1 / 17
print(ls0)
print(ls1)

fig5 = plt.figure()
plt.plot(loss0, 'g', label='JDEP')
plt.plot(loss1, 'b', label='KNN + LSTM')
plt.xlabel("Test Set ID")
plt.ylabel("MSE")
# plt.title('MSE of predicted results of the two methods (rate 0.2)')
plt.legend(loc='upper right')
plt.grid()
plt.show()
