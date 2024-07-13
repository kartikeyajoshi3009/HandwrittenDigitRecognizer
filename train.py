import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Activation, Conv2D, MaxPooling2D

dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

#Normalize the input data
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)

imageSize = 28
resized_x_train = np.array(x_train).reshape(-1, imageSize, imageSize, 1)
resized_x_test = np.array(x_test).reshape(-1, imageSize, imageSize, 1)

# Build the model
model = Sequential()

# Layer 1
model.add(Conv2D(64, (3, 3), input_shape=(imageSize, imageSize, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))

model.add(Dropout(0.2))
# Layer 3
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))

# Flatten the input
model.add(Flatten())

# Layer 3
model.add(Dense(64))
model.add(Activation("relu"))

# Layer 4
model.add(Dense(32))
model.add(Activation("relu"))

# Layer 5
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model
history = model.fit(resized_x_train, y_train, epochs=5, validation_split=0.3)

# Evaluate the model
test_loss, test_acc = model.evaluate(resized_x_test, y_test)
print("\nLoss in the test dataset= ", test_loss)
print("\nTest accuracy= ", test_acc)

# Save the model
model.save("handwritten_digit_rec.h5")

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
