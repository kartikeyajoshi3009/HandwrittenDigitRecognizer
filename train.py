import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Activation, Conv2D, MaxPooling2D

dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Normalize the input data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

imageSize = 28
resized_x_train = np.array(x_train).reshape(-1, imageSize, imageSize, 1)
resized_x_test = np.array(x_test).reshape(-1, imageSize, imageSize, 1)

model = Sequential()

# layer 1
model.add(Conv2D(64, (3, 3), input_shape=(imageSize, imageSize, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# layer 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))

model.add(Flatten())

# layer 3
model.add(Dense(64))
model.add(Activation("relu"))

# layer 4
model.add(Dense(32))
model.add(Activation("relu"))

# layer 5
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(resized_x_train, y_train, epochs=5, validation_split=0.3)

test_loss, test_acc = model.evaluate(resized_x_test, y_test)
print("\nLoss in the test dataset= ", test_loss)
print("\nTest accuracy= ", test_acc)
model.save("handwritten_digit_rec.h5")
