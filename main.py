import numpy as np
import pandas as pd 
from keras.utils import to_categorical
import math

'''data preparation
'''
#load .txt file
with open("music.txt") as f:
    a = f.read()

#convert file to array of int sequence
data = []
b = a.split('\n')
print(b)
for string in b:
    seq = string.split(' ')
    int_seq = [int(s) for s in seq]
    data.append(int_seq)
data = np.asarray(data)  #convert array to np array
# print(np.amax(data))
#tokenlize data
data_flat = []
for row in data:
    for num in row:
        data_flat.append(num)
# print(np.max(data_flat))
data_one_hot = to_categorical(data_flat)
print(data_one_hot.shape)


