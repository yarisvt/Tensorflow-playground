# pylint: disable=E1101

import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, ClassVar, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.messagebox as messagebox
import tkinter as tk


class Network:
    def __init__(self, input_shape: Tuple[int, int], layers: List[List[Any]], model=None):
        self.input_shape = input_shape
        self.layers = layers
        self.loaded_model = None
        self.activation = {'relu': tf.nn.relu, 'softmax': tf.nn.softmax}
        self.model = model
        self.load_data()

    @classmethod
    def from_model(cls, file_name: str):
        p = tf.keras.models.load_model(file_name)
        input_shape = None
        layers = []
        for layer in p.layers:
            if 'Flatten' in str(layer):
                input_shape = layer.input_shape[1:]
            elif 'Dense' in str(layer):
                layers.append([layer.units, layer.activation.__name__])
        return cls(input_shape=input_shape, layers=layers, model=tf.keras.models.load_model(file_name))

    def load_data(self):
        x_train, y_train, x_test, y_test = load_data()
        self.add_data(x_train, y_train, x_test, y_test)

    def add_data(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        for layer in self.layers:
            self.model.add(tf.keras.layers.Dense(
                layer[0], activation=self.activation.get(layer[1], '')))

    def compile_model(self, optimizer: str, loss: str, metrics: List[str]):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train_model(self, epochs: int):
        self.model.fit(self.train_data, self.train_labels, epochs=epochs)

    def check_accuracy_loss(self):
        val_loss, val_accuracy = self.model.evaluate(
            self.train_data, self.train_labels)
        print(f'loss -> {val_loss}\naccuracy -> {val_accuracy}')

    def save_model(self, file_name: str):
        self.model.save(file_name)

    def predict(self, image_nr: int) -> int:
        predictions = self.model.predict([self.test_data])
        return self.test_labels[image_nr], np.argmax(predictions[image_nr])

    def get_test_data(self):
        return self.test_data, self.test_labels

    def get_training_data(self):
        return self.train_data, self.train_labels


def draw(image: np.ndarray):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    return x_train, y_train, x_test, y_test


class GUI:
    def __init__(self, model='test.h5'):
        self.window = tk.Tk()
        self.window.title = 'Predicting images+'
        self.net = Network.from_model(model)
        self.test_data, _ = self.net.get_test_data()
        self.figure = None
        self.top_frame = None
        self.bottom_frame = None
        self.figures = []
        self.create_window(self.get_random_nr())

    def create_window(self, img_range):
        self.figure = plt.figure()
        self.top_frame = tk.Frame(self.window)
        self.bottom_frame = tk.Frame(self.window)
        i = 0
        for index in range(img_range, img_range + 9):
            ax = self.figure.add_subplot(330 + 1 + i)
            ax.imshow(self.test_data[index], cmap=plt.cm.binary)
            self.figures.append([ax, index])
            i += 1

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.top_frame)
        self.canvas.mpl_connect('button_press_event', self.handle_click)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tk.Button(self.bottom_frame, text='Get new random images',
                  command=self.new_images).pack()

        self.top_frame.pack()
        self.bottom_frame.pack()
        tk.mainloop()

    def handle_click(self, event):
        for fig in self.figures:
            if fig[0] == event.inaxes:
                label, prediction = self.net.predict(fig[1])
                messagebox.showinfo(
                    'Prediction', f'I predicted: {prediction}\nTrue number: {label}')

    def new_images(self):
        self.top_frame.destroy()
        self.bottom_frame.destroy()
        self.figures = []
        self.create_window(self.get_random_nr())

    def get_random_nr(self):
        return random.randrange(len(self.test_data) - 9)


# gui = GUI()
# gui.gui(gui.get_random_nr())


# net = Network.from_model('test.h5')
# net.load_data()
# net.predict(18, draw_image=True)
# example

#  network = Network((28, 28), [
#     [128, 'relu'],
#     [128, 'relu'],
#     [10, 'softmax']
# ])

# network.add_data(train_data=x_train, train_labels=y_train,
#                  test_data=x_test, test_labels=y_test)
# network.create_model()
# network.compile_model(
#     optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# network.train_model(epochs=5)
# network.save_model(file_name='test.h5')
