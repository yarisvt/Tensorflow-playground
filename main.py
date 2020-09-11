import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as pyplot

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
# print(predictions)

probabilities = tf.nn.softmax(predictions).numpy()
# print(probabilities)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

probability_model = keras.Sequential([
    model,
    keras.layers.Softmax()
])
print(probability_model(x_test[:5]))
