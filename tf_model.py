import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime, timedelta
from time import time
from preprocessing import preprocess, noisify
import matplotlib.pyplot as plt
import random
from epidemic_simulation import Simulation

np.random.seed(1234)


class SimulationData:
    def __init__(self, unit_time_data, prior_steps):
        self.unit_time_data = unit_time_data
        self.length_of_sim = len(unit_time_data)
        self.prior_steps = prior_steps

    def get_data_and_labels(self):
        """
        Returns data and labels for a single simulation according to specified prior_steps.
        Ensures that lengths of data and labels are some multiple of 32.
        """
        x = []
        y = []
        for i in range(self.length_of_sim - self.prior_steps):
            x.append(np.asarray(self.unit_time_data[i:i + self.prior_steps]))
            y.append(self.unit_time_data[i + self.prior_steps])

        x_to_add = np.asarray(x[-1])
        y_to_add = np.asarray(y[-1])

        while len(x) % 32 != 0:
            x.append(x_to_add)
            y.append(y_to_add)

        x = np.asarray(x)
        y = np.asarray(y)

        x_mean = x.mean()
        x = x / x_mean
        y = y / x_mean
        return x, y


class SimpleLSTM:
    def __init__(self,
                 units,
                 prior_steps,
                 loss_function='mean_squared_error',
                 learning_rate=0.001,
                 learning_rate_decay=1.0):
        self.prior_steps = prior_steps
        self.units = units
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epochs = 0
        self.history = None

        self.model = keras.Sequential([
            keras.layers.LSTM(units=units,
                              input_shape=(self.prior_steps, 4),
                              stateful=False),
            keras.layers.Dense(units=4)
        ])
        self.model.compile(loss=loss_function,
                           optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           metrics='acc')

    def scheduler(self, epoch):
        return self.learning_rate * (self.learning_rate_decay ** epoch)

    def train(self, training_sims_unit_data, validation_sims_unit_data, epochs):
        self.epochs = epochs
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for unit_data in training_sims_unit_data:
            sim_data = SimulationData(unit_data, self.prior_steps)
            x, y = sim_data.get_data_and_labels()
            x_train.extend(x)
            y_train.extend(y)
        for unit_data in validation_sims_unit_data:
            sim_data = SimulationData(unit_data, self.prior_steps)
            x, y = sim_data.get_data_and_labels()
            x_val.extend(x)
            y_val.extend(y)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)

        lr_scheduler = keras.callbacks.LearningRateScheduler(self.scheduler)

        self.history = self.model.fit(x_train, y_train,
                                      validation_data=(x_val, y_val),
                                      batch_size=32,
                                      epochs=epochs,
                                      callbacks=[lr_scheduler])

    def plot_loss(self):
        if self.history is None:
            print("Model has not been trained")
            return

        start_epoch = 5
        total_epochs = self.epochs
        history = self.history.history

        epoch_list = range(1, total_epochs + 1)[start_epoch:]

        loss = history['loss'][start_epoch:]
        val_loss = history['val_loss'][start_epoch:]

        plt.plot(epoch_list, loss, label='Training loss')
        plt.plot(epoch_list, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        if self.history is None:
            print("Model has not been trained")
            return

        start_epoch = 5
        total_epochs = self.epochs
        history = self.history.history

        epoch_list = range(1, total_epochs + 1)[start_epoch:]

        acc = history['acc'][start_epoch:]
        val_acc = history['val_acc'][start_epoch:]

        plt.plot(epoch_list, acc, label='Training accuracy')
        plt.plot(epoch_list, val_acc, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.show()

    def save(self, path):
        self.model.save(path, save_format='h5')


if __name__ == '__main__':
    EPOCHS = 30
    INITIAL_LR = 0.001
    LR_DECAY = 0.7
    TRAINING_SPLIT_RATIO = 0.9
    INITIAL_STEPS = 50
    NOISE = 0.15
    MODEL_PATH = 'models/temporal/model_51_from_prior_50_noisy_15'  # format model_x-y_from_prior_z, e.g. model_51-100_from_prior_50
    DATA_PATH = 'data.npz'

    data = np.load(DATA_PATH)
    files = data.files
    first_file = files[0]
    all_sims_unit_time_data = []
    for file in files:
        all_sims_unit_time_data.append(noisify(data[file], NOISE))

    new_model = SimpleLSTM(128,
                           INITIAL_STEPS,
                           learning_rate=INITIAL_LR,
                           learning_rate_decay=LR_DECAY,)

    training_data = all_sims_unit_time_data[:90]
    validation_data = all_sims_unit_time_data[90:]

    new_model.train(training_data, validation_data, EPOCHS)
    new_model.plot_loss()
    new_model.plot_accuracy()
    new_model.save(MODEL_PATH)
