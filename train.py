from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D
import tensorflow as tf
import cv2, os, random
import numpy as np

# # load mnist data
# (train_data, train_label), (eval_data, eval_label) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
# train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
# eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)
# train_data = train_data.astype("float32") / 255
# eval_data = eval_data.astype("float32") / 255

train_data = []
train_label = []
eval_data = []
eval_label = []

labels = []
os.chdir("./datasets/data")
for dir in os.listdir():
    labels = os.listdir("/home/amnesy/Desktop/sharedWAll/Scuola/" +\
               "stage-2021/beat-the-ai-1/datasets/data/")


for dir in os.listdir():
    os.chdir(dir)
    for file in os.listdir():
        path = "/home/amnesy/Desktop/sharedWAll/Scuola/" \
               "stage-2021/beat-the-ai-1/datasets/data/" + \
               dir + "/" + file
        data = cv2.imread(path)
        grey = cv2.cvtColor(data.copy(), cv2.COLOR_BGR2GRAY)
        a, thresh = cv2.threshold(grey, 70, 1, cv2.THRESH_BINARY_INV)
        resized_digit = cv2.resize(thresh, (28, 28))

        if random.randint(0, 3):
            train_data.append(resized_digit)
            train_label.append(labels.index(dir))
        else:
            eval_data.append(resized_digit)
            eval_label.append(labels.index(dir))
    os.chdir("../")

print(len(train_data))
print(len(eval_data))
train_data = np.array(train_data)
train_label = np.array(train_label)
eval_data = np.array(eval_data)
eval_label = np.array(eval_label)

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(200, activation="relu"))
model.add(Dense(63, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_data, train_label, batch_size=128, validation_data=(eval_data, eval_label), epochs=25)
model.save("../../mnist.model")

# dataset from https://www.kaggle.com/dhruvildave/english-handwritten-characters-dataset