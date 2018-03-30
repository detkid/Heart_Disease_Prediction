from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

batch_size = 50
num_classes = 2
epochs = 1000

data = np.loadtxt('data/reprocessed.hungarian.csv', skiprows=1)
train_data = data[:250]
test_data = data[250:]

x_train = np.delete(train_data, 2, 1)
y_train = train_data[:, 2]-1

x_test = np.delete(test_data, 2, 1)
y_test = test_data[:, 2]-1

# a = np.where(y_train != 3)

y_train[np.where(y_train != 3)[0]] = 0
y_train[np.where(y_train == 3)[0]] = 1
y_test[np.where(y_test != 3)[0]] = 0
y_test[np.where(y_test == 3)[0]] = 1

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

plt.hist(y_train)
plt.hist(y_test)
plt.show()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(13,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tb_cb = keras.callbacks.TensorBoard(log_dir="tflog_0329_2class/", histogram_freq=1)
cbks = [tb_cb]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test),
                    callbacks=cbks)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
