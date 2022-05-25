import tensorflow as tf
from tensorflow import keras
import numpy as np
from preprocessing import preprocess
import matplotlib.pyplot as plt
from epidemic_simulation import Simulation
np.random.seed(1234)


def model():
    """
    Returns an LSTM neural network as described in the code below.
    """
    model_to_return = keras.Sequential([
        keras.layers.LSTM(units=128,
                          input_shape=(x_train.shape[1], x_train.shape[2]),
                          return_sequences=False),
        keras.layers.Dense(units=256, activation=tf.nn.relu),
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dense(units=4)
    ])
    return model_to_return


def scheduler(epoch, lr):
    """
    This alters the learning rate for each epoch; it decreases by
    a factor of 0.9 each epoch so that greater accuracy can be
    achieved in training.
    """
    return lr * 0.9


if __name__ == '__main__':
    EPOCHS = 100
    INITIAL_LR = 0.001
    TRAINING_SPLIT_RATIO = 0.8
    INITIAL_STEPS = 50
    NAME = 'model_example'  # format model_x-y_from_prior_z, e.g. model_51-100_from_prior_50

    "Mean-normalizing the data helps training immensely."

    x_train, y_train, x_test, y_test = preprocess('data.npz', TRAINING_SPLIT_RATIO, INITIAL_STEPS)
    mean = x_train.mean()
    x_train, y_train, x_test, y_test = x_train/mean, y_train/mean, x_test/mean, y_test/mean

    model = model()

    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(lr=INITIAL_LR),
        metrics=['acc']
    )
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[callback])
    model.save('module_1_models/model_1')

    start_epoch = 2

    epoch_list = range(1, EPOCHS + 1)[start_epoch:]

    acc = history.history['acc'][start_epoch:]
    val_acc = history.history['val_acc'][start_epoch:]
    loss = history.history['loss'][start_epoch:]
    val_loss = history.history['val_loss'][start_epoch:]

    plt.plot(epoch_list, acc, label='Training accuracy')
    plt.plot(epoch_list, val_acc, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epoch_list, loss, label='Training loss')
    plt.plot(epoch_list, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
