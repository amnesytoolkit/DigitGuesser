import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D

# load mnist data
(train_data, train_label), (eval_data, eval_label) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)
train_data = train_data.astype("float32") / 255
eval_data = eval_data.astype("float32") / 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_data, train_label, batch_size=128, validation_data=(eval_data, eval_label), epochs=5)

model.save("mnist.model")