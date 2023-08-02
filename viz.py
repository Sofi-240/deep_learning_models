from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K

matplotlib.use("Qt5Agg")


def show(image, ax):
    if tf.is_tensor(image):
        image = image.numpy()
    if image.max() > 1:
        image = image.astype('uint8')
    if len(image.shape) == 2 or image.shape[-1] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)


def show_images(images, subplot_y=None, subplot_x=None):
    assert issubclass(type(images), (np.ndarray, tf.Tensor, list, tuple))
    subplot_x = min([len(images), 4]) if subplot_x is None else subplot_x
    subplot_y = len(images) // subplot_x if subplot_y is None else subplot_y

    fig, _ = plt.subplots(subplot_x, subplot_y, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0, hspace=0.05)

    for i in range(min([subplot_x * subplot_y, len(images)])):
        show(images[i], fig.axes[i])


def learning_rate_viz(lr_schedule, data_size=1000, epochs=50, seed=None):
    class TapeLearningRate(keras.callbacks.Callback):
        def __init__(self):
            super(TapeLearningRate, self).__init__()
            self.lr = []

        def on_batch_begin(self, batch, logs=None):
            lr = float(
                K.get_value(self.model.optimizer.lr)
            )
            self.lr.append(lr)

    seed = seed if seed is not None else 0
    np.random.seed(seed)
    x_data = 2 * np.random.random(size=(data_size, 1))
    y_data = np.random.normal(loc=x_data ** 2, scale=0.05)

    model = keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='tanh', input_dim=1),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    callback = [TapeLearningRate()]
    if issubclass(type(lr_schedule), keras.callbacks.Callback):
        callback.append(lr_schedule)
        optimizer = keras.optimizers.Adam()
    elif issubclass(type(lr_schedule), keras.optimizers.schedules.LearningRateSchedule):
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        return

    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(x_data, y_data, epochs=epochs, callbacks=callback)

    _, ax = plt.subplots(1, 2)
    ax[0].set(xlabel='Epoch', ylabel='Loss')
    ax[0].plot(history.history['loss'])
    ax[1].set(xlabel='step', ylabel='LR')
    ax[1].plot(callback[0].lr)
