
import pickle
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
import os
import h5py

# Reload the data
pickle_file = 'data2.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_d']
    y_train = pickle_data['train_l']
    X_valid = pickle_data['valid_d']
    y_valid = pickle_data['valid_l']
    X_test = pickle_data['test_d']
    y_test = pickle_data['test_l']
    del pickle_data  


# normalize
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test  = X_test.astype('float32')
X_train = X_train / 255 - 0.5
X_valid = X_valid / 255 - 0.5
X_test  = X_test /  255 - 0.5

# img shape
img_shape = X_train.shape[1:]
print('image shape:', img_shape)

# set hyperparameters and print out the summary
np.random.seed(2017)
batch_size = 64 
epochs = 15 

pool_size = (2, 2)
kernel_size = (5, 5)

# initiate the model
model = Sequential()

#first convo layer
model.add(Convolution2D(16, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=img_shape))
# activate
model.add(Activation('relu'))
# 2nd convo layer
model.add(Convolution2D(32, kernel_size[0], kernel_size[1],subsample=(2, 2)))
# activate
model.add(Activation('relu'))
# third and thickest convo layer
model.add(Convolution2D(64, kernel_size[0], kernel_size[1],subsample=(2, 2)))
# activate
model.add(Activation('relu'))
# max pooling
model.add(MaxPooling2D(pool_size=pool_size))
# dropout
model.add(Dropout(0.5))
# flatten
model.add(Flatten())
# activate
model.add(Activation('relu'))
# fully connected1
model.add(Dense(512))
# activate
model.add(Activation('relu'))
# fully connected2
model.add(Dense(16))
# activate
model.add(Activation('relu'))
# dropout
model.add(Dropout(0.5))
# activate
model.add(Activation('relu'))
# output 1 - as we just need a single number float
model.add(Dense(1))

# print summary
model.summary()

# compile
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

# train
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=epochs,
                    verbose=1, validation_data=(X_valid, y_valid))
evalu = model.evaluate(X_test, y_test, verbose=0)
print('Score:', evalu[0])
print('Accuracy:', evalu[1])



json_string = model.to_json()

with open('model.json', 'w') as outfile:
	json.dump(json_string, outfile)
	model.save_weights('./model.h5')
	print("Model saved")