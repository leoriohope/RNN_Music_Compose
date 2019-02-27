import numpy as np
import pandas as pd 
import math
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

'''data preparation
'''
def viewData(fileName):
    '''
    Input:
    fileName: str
    Output:
    length: int  #length of the music samle
    total: int   #totoal number of unit
    '''
    with open(fileName, 'r') as f:
        length = 0
        total = 0
        for line in f:
            length += 1
            total += len(line.split(" "))
    return length, total

length, total = viewData('music.txt')
# print(length)
# print(total // length)
n_X = length
len_X = total // length

def loadData(fileName):
    '''
    Input:
        filename : str
    Output:
        int_data: nparray (703)
    '''
    int_data = []
    with open(fileName, 'r') as f:
        for line in f:
            int_data.append([int(num) for num in line.split(' ')])
    return int_data

def flattenData(data):
    '''
    convert the data to a long array for tokenize
    Input:
    data: list of list
    Output:
    res: a long list of data
    '''
    res = []
    for lst in data:
        for num in lst:
            res.append(num)
    return res

int_data = loadData('music.txt')
# print(len(int_data))
flatten_data = flattenData(int_data)
flatten_data = np.asarray(flatten_data) # (1553149, ) nparray
flatten_data = flatten_data[:1552927]
data_token = to_categorical(flatten_data)
data_token = np.reshape(data_token, (703, 2209, 114))
X = data_token
print(X.shape)
Y = np.asarray([np.append(lst[-1:], lst[: -1]) for lst in data_token])
Y = np.reshape(Y, (2209, 703, 114))
print(Y.shape)
# print(data_token[0][0])
# flatten_data = np.reshape(flatten_data, (703, 2209))

n_a = 64
n_values = 114
reshapor = Reshape((1, 114))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D

def djmodel(Tx, n_a, n_values):
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []
    for t in range(Tx):
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model

model = djmodel(Tx = 2209 , n_a = 64, n_values = 114)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = 703
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    outputs = []
    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(one_hot)(out)
        inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    
    return inference_model

inference_model = music_inference_model(LSTM_cell, densor, n_values = 114, n_a = 64, Ty = 50)
x_initializer = np.zeros((1, 1, 114))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis=-1)
    results = to_categorical(indices, num_classes=78)    
    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)





