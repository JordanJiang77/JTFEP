import numpy as np
from tensorflow.python.keras import layers, models
import pandas as pd
import random
from dataUSE import dataDirect0
from misspredict import Improvedlstm
import matplotlib.pyplot as plt
from dataUSE import Datadir0

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt, a_prev, c_prev):
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros((n_a + n_y, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = np.multiply(ot, np.tanh(c_next))

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = np.dot(Wy, a_next) + by

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
    for i in range(33):
        loss += (y[i]-y_hat[i])**2
    return loss
# fix random seed for reproducibility
np.random.seed(7)

# load data
dataset = Datadir0
# print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.75)
test = dataset[train_size:train_size+33]
# print(train)
# print(test)
t = list_split(test, 33)
t = np.array(t)
Testbatch = np.reshape(t, (t.shape[0], t.shape[1], 1))
n_x, m, T_x = Testbatch.shape
a_prev = np.zeros((16, 33))
c_next = np.zeros((16, m))

losspredit = []
epoches  = 1
for ve in range(epoches):
    a = random.randint(0, 20)
    rate = 0.1
    nummiss = int(33 * rate)
    train = dataset[0:train_size]
    x = list_split(train, 33)
    m, n = np.shape(x)
    for i in range(m):
        for j in range(a, a + nummiss):
            x[i][j] = 'x'
    Comdata = []
    Missdata = []
    for i in range(0, m):
        for j in range(33):
            if x[i][j] == 'x':
                Comdata.append(0)
            else:
                Comdata.append(train[i])

    for i in range(0, m):
        for j in range(33):
            if x[i][j] == 'x':
                a = random.randrange(0, 20)
                Missdata.append(a)
            else:
                Missdata.append(0)
    x = list_split(Comdata, 33)
    sigema = list_split(Missdata, 33)

    x = np.array(x)
    y = np.array(x)
    sigema = np.array(sigema)
    x_train = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y_train = np.reshape(y, (y.shape[0], y.shape[1], 1))
    sigema = np.reshape(sigema, (sigema.shape[0], sigema.shape[1], 1))
    # train the model and realize test
    char_to_idx = x_train
    idx_to_char = y_train

    lstm = Improvedlstm(sigema, char_to_idx, idx_to_char)
    loss, parame = lstm.optimize(sigema, char_to_idx, idx_to_char)
    loss = loss
    print(loss)
    # print(parame)

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

    for t in range(T_x):
        a_next, c_next, yt, cache, = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt=Testbatch[:, :, t],
                                                       a_prev=a_prev, c_prev=c_next)
        # print(yt)
    x_batch = Testbatch
    y_batch = Testbatch
    for t in range(T_x):
        x = Testbatch[:, :, t]
    a, y_pred, c, caches = lstm_forward(Wy, x_batch, a_prev)
    # print(y_pred[2])
    T, m, n = y_pred.shape
    vapredict = y_pred[3]
    predictiona = []
    for i in range(33):
        predictiona.append(vapredict[i].tolist())
    prediction1 = sum(predictiona, [])
    print(prediction1)
    loss1 = compute_loss(y_hat=prediction1, y=test)
    print(loss1)
    a = random.randint(0, 20)
    rate = 0.2
    nummiss = int(33 * rate)
    train = dataset[0:train_size]
    x = list_split(train, 33)
    m, n = np.shape(x)
    for i in range(m):
        for j in range(a, a + nummiss):
            x[i][j] = 'x'
    Comdata = []
    Missdata = []
    for i in range(0, m):
        for j in range(33):
            if x[i][j] == 'x':
                Comdata.append(0)
            else:
                Comdata.append(train[i])

    for i in range(0, m):
        for j in range(33):
            if x[i][j] == 'x':
                a = random.randrange(0, 20)
                Missdata.append(a)
            else:
                Missdata.append(0)
    x = list_split(Comdata, 33)
    sigema = list_split(Missdata, 33)

    x = np.array(x)
    y = np.array(x)
    sigema = np.array(sigema)
    x_train = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y_train = np.reshape(y, (y.shape[0], y.shape[1], 1))
    sigema = np.reshape(sigema, (sigema.shape[0], sigema.shape[1], 1))
    # train the model and realize test
    char_to_idx = x_train
    idx_to_char = y_train

    lstm = Improvedlstm(sigema, char_to_idx, idx_to_char)
    loss, parame = lstm.optimize(sigema, char_to_idx, idx_to_char)
    loss = loss
    print(loss)
    # print(parame)

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

    for t in range(T_x):
        a_next, c_next, yt, cache, = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt=Testbatch[:, :, t],
                                                       a_prev=a_prev, c_prev=c_next)
        # print(yt)
    x_batch = Testbatch
    y_batch = Testbatch
    for t in range(T_x):
        x = Testbatch[:, :, t]
    a, y_pred, c, caches = lstm_forward(Wy, x_batch, a_prev)
    # print(y_pred[2])
    T, m, n = y_pred.shape
    vapredict = y_pred[3]
    predictionb = []
    for i in range(33):
        predictionb.append(vapredict[i].tolist())
    prediction2 = sum(predictionb, [])
    print(prediction2)
    loss2 = compute_loss(y_hat=prediction2, y=test)
    print(loss2)
    a = random.randint(0, 20)
    rate = 0.3
    nummiss = int(33 * rate)
    train = dataset[0:train_size]
    x = list_split(train, 33)
    m, n = np.shape(x)
    for i in range(m):
        for j in range(a, a + nummiss):
            x[i][j] = 'x'
    Comdata = []
    Missdata = []
    for i in range(0, m):
        for j in range(33):
            if x[i][j] == 'x':
                Comdata.append(0)
            else:
                Comdata.append(train[i])

    for i in range(0, m):
        for j in range(33):
            if x[i][j] == 'x':
                a = random.randrange(0, 20)
                Missdata.append(a)
            else:
                Missdata.append(0)
    x = list_split(Comdata, 33)
    sigema = list_split(Missdata, 33)

    x = np.array(x)
    y = np.array(x)
    sigema = np.array(sigema)
    x_train = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y_train = np.reshape(y, (y.shape[0], y.shape[1], 1))
    sigema = np.reshape(sigema, (sigema.shape[0], sigema.shape[1], 1))
    # train the model and realize test
    char_to_idx = x_train
    idx_to_char = y_train

    lstm = Improvedlstm(sigema, char_to_idx, idx_to_char)
    loss, parame = lstm.optimize(sigema, char_to_idx, idx_to_char)
    loss = loss
    print(loss)
    # print(parame)

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

    for t in range(T_x):
        a_next, c_next, yt, cache, = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt=Testbatch[:, :, t],
                                                       a_prev=a_prev, c_prev=c_next)
        # print(yt)
    x_batch = Testbatch
    y_batch = Testbatch
    for t in range(T_x):
        x = Testbatch[:, :, t]
    a, y_pred, c, caches = lstm_forward(Wy, x_batch, a_prev)
    # print(y_pred[2])
    T, m, n = y_pred.shape
    vapredict = y_pred[3]
    predictionc = []
    for i in range(33):
        predictionc.append(vapredict[i].tolist())
    prediction3 = sum(predictionc, [])
    print(prediction3)
    loss3 = compute_loss(y_hat=prediction3, y=test)
    print(loss3)
    # plot predicti
    fig = plt.figure()
    plt.plot(prediction1, 'r', label='Predicted flow missing rate 0.1')
    plt.plot(prediction2, 'b', label='Predicted flow missing rate 0.2')
    plt.plot(prediction3, 'c', label='Predicted flow missing rate 0.3')
    plt.plot(test, 'g', label='Real flow')
    plt.xlabel("Sensor ID")
    plt.ylabel("Traffic Flow (vehicles/min)")
    plt.title('Traffic data of different detectors at one time')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    losspredit.append(loss1)
    print(losspredit)
meanlosspredit = 0
for i in range(epoches):
    meanlosspredit += losspredit[i]
meanlosspredit = meanlosspredit / epoches
print(meanlosspredit)