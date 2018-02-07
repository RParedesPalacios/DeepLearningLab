from __future__ import print_function

import keras
from keras.callbacks import LearningRateScheduler as LRS

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import SGD



batch_size = 100
num_classes = 10
epochs = 75

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize [0..255]-->[0..1]
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


sgd=SGD(lr=0.1, decay=0.0, momentum=0.0)

def scheduler(epoch):
    if epoch < 2:
        return .1
    elif epoch < 50:
        return 0.01
    else:
        return 0.001

set_lr = LRS(scheduler)


model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[set_lr])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
