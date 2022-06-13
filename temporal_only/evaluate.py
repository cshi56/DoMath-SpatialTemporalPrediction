import numpy as np
from tensorflow import keras
from collections import OrderedDict
import contextlib
import matplotlib.pyplot as plt


def mse(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    diff = a - b
    return sum(diff ** 2) / len(diff)


def evaluate(val_sims, nn_model, input_length, length_of_forecast, stride):
    ret = OrderedDict()
    mses = []
    for seir_data in val_sims:
        n = sum(seir_data[0])
        seir_data = seir_data / n
        sim_length = len(seir_data)

        for t in range(0, sim_length - input_length - length_of_forecast, stride):
            initial_data = seir_data[t:t + input_length]
            initial_data = np.expand_dims(initial_data, 0)
            y_hat = []

            for h in range(length_of_forecast):
                next_data = np.asarray(nn_model.predict_step(initial_data))[0]
                y_hat.append(next_data)
                initial_data = initial_data[:, 1:]
                initial_data = np.append(initial_data, [[next_data]], axis=1)

            y_hat = np.asarray(y_hat)
            y_actual = seir_data[t + input_length:t + input_length + length_of_forecast]
            mean_se = mse(y_actual, y_hat)
            mses.append(mean_se)

    mses = np.asarray(mses)
    mses_sorted = np.sort(mses)
    average_mse = mses.mean()
    std = mses.std()

    nn_model.summary()
    ret['Number of validation simulations'] = len(val_sims)
    ret['Stride between inputs'] = stride
    ret['Length of inputs'] = input_length
    ret['Total evaluation distance'] = length_of_forecast
    ret['Average mean squared error'] = average_mse
    ret['Standard deviation of mean squared error'] = std
    ret['Sorted MSE values'] = mses_sorted

    [print(key, ':', value) for key, value in ret.items()]
    print('\n')


if __name__ == '__main__':
    FILE_PATH = '../../data/data_200_sims.npz'
    TRAIN_NUM = 100
    VAL_NUM = 50
    TEST_NUM = 50
    EVALUATION_DIST = 50
    STRIDE = 50

    MODEL_PATH = 'models/m5'
    PRIOR_STEPS = 50
    FUTURE_STEPS = 1

    all_sims = np.load(FILE_PATH)
    val_file_names = all_sims.files[TRAIN_NUM:TRAIN_NUM + VAL_NUM]
    validation_sims = []

    for file_name in val_file_names:
        validation_sims.append(all_sims[file_name])

    model = keras.models.load_model(MODEL_PATH)

    file_path = 'model_evaluations.txt'
    with open(file_path, "a") as f:
        with contextlib.redirect_stdout(f):
            evaluate(validation_sims, model, PRIOR_STEPS, EVALUATION_DIST, STRIDE)
