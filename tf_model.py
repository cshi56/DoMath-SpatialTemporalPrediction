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
        Returns data and labels according to specified prior_steps.
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
        self.current_epoch = 1
        self.total_epochs = 0
        self.history = {'training_acc': [],
                        'validation_acc': [],
                        'training_loss': [],
                        'validation_loss': []}

        self.model = keras.Sequential([
            keras.layers.LSTM(units=units,
                              input_shape=(self.prior_steps, 4)),
            keras.layers.Dense(units=4)
        ])
        self.model.compile(loss=loss_function,
                           optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           metrics='acc')
        self.model.save('model_buffer', save_format='h5')

    def train_single_sim_epoch(self, simulation_unit_time_data):
        """
        Training the model on data from a single simulation.
        """
        sim_data = SimulationData(simulation_unit_time_data, self.prior_steps)
        x, y = sim_data.get_data_and_labels()
        self.model.fit(x, y, epochs=1, verbose=0)

    def train_multiple_sims_epoch(self, list_of_sims_unit_time_data):
        """
        Trains the model on all simulations in list_of_sims_unit_time_data.
        """

        "The following three lines are included for printing purposes."
        current_simulation = 0
        equal_signs_so_far = 0
        print(f'Epoch ({self.current_epoch:02d}/' + str(self.total_epochs) + '): ', end='')

        for simulation_unit_time_data in list_of_sims_unit_time_data:
            self.train_single_sim_epoch(simulation_unit_time_data)

            "The remainder of this block is also included for printing purposes."
            fraction_done = current_simulation / len(list_of_sims_unit_time_data)
            equal_signs_to_print = int(30 * fraction_done)
            for _ in range(equal_signs_to_print - equal_signs_so_far):
                print('=', end='')
                equal_signs_so_far += 1
            current_simulation += 1
        for _ in range(30 - equal_signs_so_far):
            print('=', end='')
        print(' ', end='')

        """We save the model to a buffer so that in the next epoch we can reload
        with a different learning rate."""
        self.model.save('model_buffer', save_format='h5')
        self.learning_rate *= self.learning_rate_decay

    def evaluate(self, validation_data_list_of_unit_time_sims):
        """
        Evaluating model's validation loss and accuracy and printing results.
        """
        total_loss = 0
        total_accuracy = 0
        for simulation_unit_data in validation_data_list_of_unit_time_sims:
            single_validation_datum = SimulationData(simulation_unit_data, self.prior_steps)
            actual_x, actual_y = single_validation_datum.get_data_and_labels()
            metrics = self.model.evaluate(actual_x, actual_y, verbose=0)
            sim_loss = metrics[0]
            sim_acc = metrics[1]
            total_loss += sim_loss
            total_accuracy += sim_acc
        average_loss = total_loss / len(validation_data_list_of_unit_time_sims)
        average_accuracy = total_accuracy / len(validation_data_list_of_unit_time_sims)

        print('Average validation loss: ' + str(average_loss) +
              ' - Average validation accuracy: ' + str(average_accuracy))

    def train_multiple_epochs(self, list_of_sims_unit_time_data, validation_sims, epochs):
        """
        Trains the model for the specified number of epochs; shuffles order of
        simulations between epochs.
        """
        self.total_epochs = epochs
        for _ in range(epochs):
            t1 = time()

            self.model = keras.models.load_model('model_buffer')
            self.model.compile(loss=self.loss_function,
                               optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                               metrics='acc')
            self.train_multiple_sims_epoch(list_of_sims_unit_time_data)
            random.shuffle(list_of_sims_unit_time_data)
            print('Done. ', end='')

            self.evaluate(validation_sims)

            t2 = time()
            epoch_time = t2 - t1
            time_left = epoch_time * (self.total_epochs - 1)
            if self.current_epoch == 1:
                print("\nEstimated time at end of training: ", end='')
                print((datetime.now() + timedelta(seconds=time_left)).strftime('%H:%M:%S'))
                print('')

            self.current_epoch += 1

    def train(self, training_sims_unit_data, validation_sims_unit_data, epochs):
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

        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=epochs)

    def save(self, path):
        self.model.save(path, save_format='h5')


if __name__ == '__main__':
    EPOCHS = 100
    INITIAL_LR = 0.001
    LR_DECAY = 0.9
    TRAINING_SPLIT_RATIO = 0.9
    INITIAL_STEPS = 50
    NAME = 'model_51_from_prior_50_v5'  # format model_x-y_from_prior_z, e.g. model_51-100_from_prior_50

    data = np.load('data.npz')
    files = data.files
    first_file = files[0]
    all_sims_unit_time_data = []
    for file in files:
        all_sims_unit_time_data.append(data[file])

    new_model = SimpleLSTM(128,
                           INITIAL_STEPS,
                           learning_rate=INITIAL_LR,
                           learning_rate_decay=LR_DECAY,)
    training_data = all_sims_unit_time_data[:80]
    validation_data = all_sims_unit_time_data[80:]
    new_model.train(training_data, validation_data, EPOCHS)
    new_model.save('module_1_models/' + NAME)

"""
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
"""
