from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
num_classes = 5
epochs = 2000

hungarian_data = np.loadtxt('data/reprocessed.hungarian.csv', skiprows=1)
va_data = np.loadtxt('data/processed.va.csv', skiprows=1, delimiter=',')
switzerland_data = np.loadtxt(
    'data/processed.switzerland.csv', skiprows=1, delimiter=',')
cleveland_data = np.loadtxt(
    'data/processed.cleveland.csv', skiprows=1, delimiter=',')

data = np.vstack((hungarian_data, va_data, switzerland_data, cleveland_data))

# データ並び替え
test_ids = np.random.permutation(len(data))
data = data[test_ids]

train_data = data[:800]
test_data = data[800:]

x_train = np.delete(train_data, 13, 1)
y_train = train_data[:, 13]

x_test = np.delete(test_data, 13, 1)
y_test = test_data[:, 13]

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

tb_cb = keras.callbacks.TensorBoard(log_dir="log/tflog_0330_4class/", histogram_freq=1)
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
