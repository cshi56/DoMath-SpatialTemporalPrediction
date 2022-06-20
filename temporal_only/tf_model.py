import keras as keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import LSTM, Dense
import numpy as np
from preprocessing import preprocess, noisify
import matplotlib.pyplot as plt

NEG_PARAM = 1E-4
OVER_ONES_PARAM = 1E-4

np.random.seed(1234)
tf.random.set_seed(1234)


class CustomLoss(keras.losses.Loss):
    def __init__(self, neg_param, over_ones_param):
        self.neg_param = neg_param
        self.over_ones_param = over_ones_param
        super().__init__()

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        zeros = tf.zeros_like(y_pred)
        negs = tf.math.minimum(y_pred, zeros)
        neg_mean = tf.reduce_mean(negs)

        ones = tf.ones_like(y_pred)
        pred_less_one = y_pred - ones
        over_ones = tf.math.maximum(pred_less_one, zeros)
        over_ones_mean = tf.reduce_mean(over_ones)
        return mse - (self.neg_param * neg_mean) + (self.over_ones_param * over_ones_mean)


class SimpleLSTM:
    def __init__(self,
                 lstm_units_list,
                 dense_units_list,
                 prior_steps,
                 future_steps,
                 loss_function=CustomLoss(NEG_PARAM, OVER_ONES_PARAM),
                 learning_rate=0.001,
                 learning_rate_decay=1.0):
        self.prior_steps = prior_steps
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epochs = 0
        self.history = None

        self.model = keras.Sequential()

        if len(lstm_units_list) == 1:
            self.model.add(LSTM(units=lstm_units_list[0],
                                input_shape=(self.prior_steps, 4),
                                return_sequences=False,
                                stateful=False))

        else:
            self.model.add(LSTM(units=lstm_units_list[0],
                                input_shape=(self.prior_steps, 4),
                                return_sequences=True,
                                stateful=False))
            for lstm_units in lstm_units_list[1:-1]:
                self.model.add(LSTM(units=lstm_units,
                                    return_sequences=True,
                                    stateful=False))
            self.model.add(LSTM(units=lstm_units_list[-1],
                                return_sequences=False,
                                stateful=False))

        for dense_units in dense_units_list:
            self.model.add(Dense(units=dense_units, activation=tf.nn.relu))
        self.model.add(Dense(units=4*future_steps))

        self.model.compile(loss=loss_function,
                           optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                                           beta_1=0.9,
                                                           beta_2=0.999,
                                                           epsilon=1E-7,
                                                           amsgrad=False),
                           metrics='acc')
        self.model.build(input_shape=(self.prior_steps, 4))

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

        for start_epoch in [5, 70, 100, 120]:

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

    def summary(self):
        self.model.summary()

    def save(self, path):
        self.model.save(path, save_format='h5')


if __name__ == '__main__':
    PRIOR_STEPS = 50
    LSTM_UNITS_LISTS = [[64], [128], [128], [128], [128, 64], [128, 64]]
    DENSE_UNITS_LISTS = [[], [], [64], [128, 64], [64], [128, 64]]
    FUTURE_STEPS_LIST = [2, 5, 10]

    EPOCHS = 150
    BATCH_SIZE = 32
    INITIAL_LR = 0.001
    LR_DECAY = 0.95
    LOSS = CustomLoss(NEG_PARAM, OVER_ONES_PARAM)
    TRAIN_NUM = 100
    VAL_NUM = 50
    TEST_NUM = 50
    STRIDE = 15
    STRIDES_LIST = [8, 4, 2]
    NOISE = 0.15
    DATA_PATH = '../../data/data_200_sims.npz'

    for index in range(len(LSTM_UNITS_LISTS)):
        FUTURE_STEPS = 1
        LSTM_UNITS_LIST = LSTM_UNITS_LISTS[index]
        DENSE_UNITS_LIST = DENSE_UNITS_LISTS[index]
        train_x, train_y, val_x, val_y, test_x, test_y = preprocess(DATA_PATH,
                                                                    TRAIN_NUM,
                                                                    VAL_NUM,
                                                                    TEST_NUM,
                                                                    PRIOR_STEPS,
                                                                    FUTURE_STEPS,
                                                                    stride=STRIDE)
        MODEL_PATH = 'models/m' + str(index + 1)
        new_model = SimpleLSTM(LSTM_UNITS_LIST,
                               DENSE_UNITS_LIST,
                               PRIOR_STEPS,
                               FUTURE_STEPS,
                               loss_function=LOSS,
                               learning_rate=INITIAL_LR,
                               learning_rate_decay=LR_DECAY)

        new_model.summary()

        new_model.train(train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCHS)
        new_model.save(MODEL_PATH)

    for index in range(len(FUTURE_STEPS_LIST)):
        LSTM_UNITS_LIST = LSTM_UNITS_LISTS[-1]
        DENSE_UNITS_LIST = DENSE_UNITS_LISTS[-1]
        FUTURE_STEPS = FUTURE_STEPS_LIST[index]
        train_x, train_y, val_x, val_y, test_x, test_y = preprocess(DATA_PATH,
                                                                    TRAIN_NUM,
                                                                    VAL_NUM,
                                                                    TEST_NUM,
                                                                    PRIOR_STEPS,
                                                                    FUTURE_STEPS,
                                                                    stride=STRIDE)
        MODEL_PATH = 'models/m' + str(index + 7)
        new_model = SimpleLSTM(LSTM_UNITS_LIST,
                               DENSE_UNITS_LIST,
                               PRIOR_STEPS,
                               FUTURE_STEPS,
                               loss_function=LOSS,
                               learning_rate=INITIAL_LR,
                               learning_rate_decay=LR_DECAY)

        new_model.summary()

        new_model.train(train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCHS)
        new_model.save(MODEL_PATH)

    for index in range(len(STRIDES_LIST)):
        FUTURE_STEPS = 1
        LSTM_UNITS_LIST = LSTM_UNITS_LISTS[-1]
        DENSE_UNITS_LIST = DENSE_UNITS_LISTS[-1]
        STRIDE = STRIDES_LIST[index]
        train_x, train_y, val_x, val_y, test_x, test_y = preprocess(DATA_PATH,
                                                                    TRAIN_NUM,
                                                                    VAL_NUM,
                                                                    TEST_NUM,
                                                                    PRIOR_STEPS,
                                                                    FUTURE_STEPS,
                                                                    stride=STRIDE)
        MODEL_PATH = 'models/m' + str(index + 10)
        new_model = SimpleLSTM(LSTM_UNITS_LIST,
                               DENSE_UNITS_LIST,
                               PRIOR_STEPS,
                               FUTURE_STEPS,
                               loss_function=LOSS,
                               learning_rate=INITIAL_LR,
                               learning_rate_decay=LR_DECAY)

        new_model.summary()

        new_model.train(train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCHS)
        new_model.save(MODEL_PATH)
