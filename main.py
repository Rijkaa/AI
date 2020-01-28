import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras.layers.recurrent import LSTM, GRU
from keras import regularizers
from keras.utils import np_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

datalx = pd.read_csv('./data/ncaa_learn.csv', sep=';')
dataly = pd.read_csv('./data/nba_learn.csv', sep=';')
datatx = pd.read_csv('./data/ncaa_test.csv', sep=';')
dataty = pd.read_csv('./data/nba_test.csv', sep=';')
datalxv = datalx.ix[:, 'efg'].tolist()
datalxv2 = datalx.ix[:, 'ts'].tolist()
datalyv = dataly.ix[:, 'efg'].tolist()
datalyv2 = datalx.ix[:, 'ts'].tolist()
datatxv = datatx.ix[:, 'efg'].tolist()
datatxv2 = datalx.ix[:, 'ts'].tolist()
datatyv = dataty.ix[:, 'ts'].tolist()
datatyv2 = datalx.ix[:, 'ts'].tolist()
lxv = np.ones((100, 2), dtype=np.complex_)
lyv = np.ones((100, 2), dtype=np.complex_)
txv = np.ones((20, 2), dtype=np.complex_)
tyv = np.ones((20, 2), dtype=np.complex_)
for i in range(100):
    lxv[i][0] = datalxv[i]
    lxv[i][1] = datalxv2[i]
    lyv[i][0] = datalyv[i]
    lyv[i][1] = datalyv2[i]
for i in range(20):
    txv[i][0] = datatxv[i]
    txv[i][1] = datatxv2[i]
    tyv[i][0] = datatyv[i]
    tyv[i][1] = datatyv2[i]


NB_EPOCH = 100
BATCH_SIZE = 1
VERBOSE = 1
NB_CLASSES = 2
OPTIMIZER = Adam
N_HIDDEN = 8
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

X_train = lxv
Y_train = lyv
X_test = txv
Y_test = tyv

#model = Sequential()
#model.add(Dense(N_HIDDEN, input_dim=2))
#model.add(Activation('relu'))
#model.add(Dropout(DROPOUT))
#model.add(Dense(N_HIDDEN))
#model.add(Activation('relu'))
#model.add(Dropout(DROPOUT))
#model.add(Dense(NB_CLASSES))
#model.add(Activation('softmax'))
#model.summary()
#model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
#checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
#history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
#history = model.fit(X_train, Y_train,
#          nb_epoch = 1,
#          batch_size = 128,
#          verbose=1,
#          validation_data=(X_test, Y_test),
#          callbacks=[reduce_lr, checkpointer],
#          shuffle=True)
#score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
#print("Test score:", score[0])
#print("Test accuracy:", score[1])




#model = Sequential()
#model.add(Dense(8))
   #             activity_regularizer=regularizers.l2(0.01)))
#model.add(BatchNormalization())
#model.add(LeakyReLU())

#model.add(Dropout(0.25))
#model.add(Dense(8))
       #         activity_regularizer=regularizers.l2(0.01)))
#model.add(BatchNormalization())
#model.add(LeakyReLU())
#model.add(Dense(2))
#model.add(Activation('softmax'))

#opt = Nadam(lr=0.001)

#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
#checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
#model.compile(optimizer=opt,
     #         loss='categorical_crossentropy',
      #        metrics=['accuracy'])
#nb_epoch = 100
#history = model.fit(X_train, Y_train,
#          validation_split=0.33,
#          epochs = nb_epoch,
#          batch_size = 100,
#          verbose=1,
#          validation_data=(X_test, Y_test))
#          callbacks=[reduce_lr, checkpointer])

model = Sequential()
model.add(Dense(N_HIDDEN))
model.add(Activation('softmax'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('softmax'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.compile(loss='msle', optimizer=Nadam(lr=0.001), metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
#score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
#print("Test score:", score[0])
#print("Test accuracy:", score[1])


plt.figure()
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
fig, ax = plt.subplots()
plt.show()
fig.savefig('loss')

plt.figure()
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
plt.legend(['train', 'test'], loc='best')
fig, ax = plt.subplots()
plt.show()
fig.savefig('acc')

plt.figure()
plt.plot(history.history['time'])
plt.title('time')
plt.ylabel('time')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
