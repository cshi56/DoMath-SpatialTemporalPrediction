import tensorflow as tf
from tensorflow import keras
import numpy as np
from preprocessing import preprocess, noisify
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.random.set_seed(1234)


class SimpleLSTM:
    def __init__(self,
                 units,
                 prior_steps,
                 future_steps,
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
                              return_sequences=True,
                              stateful=False),
            keras.layers.LSTM(units=128),
            keras.layers.Dense(units=128),
            keras.layers.Dense(units=4*future_steps)
        ])
        self.model.compile(loss=loss_function,
                           optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                                           beta_1=0.9,
                                                           beta_2=0.999,
                                                           epsilon=1E-7,
                                                           amsgrad=False),
                           metrics='acc')

    def scheduler(self, epoch):
        return self.learning_rate * (self.learning_rate_decay ** epoch)

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        self.epochs = epochs

        lr_scheduler = keras.callbacks.LearningRateScheduler(self.scheduler)

        self.history = self.model.fit(x_train, y_train,
                                      validation_data=(x_val, y_val),
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      callbacks=[lr_scheduler],
                                      shuffle=True)

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

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path, save_format='h5')


if __name__ == '__main__':
    PRIOR_STEPS = 50
    UNITS = 256
    FUTURE_STEPS = 1

    EPOCHS = 100
    BATCH_SIZE = 32
    INITIAL_LR = 0.001
    LR_DECAY = 0.95
    LOSS = tf.keras.losses.MeanSquaredError('sum_over_batch_size')
    TRAIN_NUM = 100
    VAL_NUM = 50
    TEST_NUM = 50
    STRIDE = 1
    NOISE = 0.15
    MODEL_PATH = 'models/m5'
    DATA_PATH = '../../data/data_200_sims.npz'

    prior_steps_list = [5, 20, 50]
    units_list = [32, 64, 128]
    future_steps_list = [1, 4, 16]

    train_x, train_y, val_x, val_y, test_x, test_y = preprocess(DATA_PATH,
                                                                TRAIN_NUM,
                                                                VAL_NUM,
                                                                TEST_NUM,
                                                                PRIOR_STEPS,
                                                                FUTURE_STEPS,
                                                                stride=STRIDE)

    new_model = SimpleLSTM(UNITS,
                           PRIOR_STEPS,
                           FUTURE_STEPS,
                           loss_function=LOSS,
                           learning_rate=INITIAL_LR,
                           learning_rate_decay=LR_DECAY)

    print(new_model.summary())

    new_model.train(train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCHS)
    new_model.plot_loss()
    new_model.plot_accuracy()
    new_model.save(MODEL_PATH)
