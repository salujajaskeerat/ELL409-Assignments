import tensorflow as tf

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Loading data
data = tf.keras.datasets.fashion_mnist


#
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# plt.imshow(train_images[1], cmap=plt.cm.binary)
# plt.show()


train_images = train_images/255.0
test_images = test_images/255.0


# Creating model

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax'),
                          ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=20)
