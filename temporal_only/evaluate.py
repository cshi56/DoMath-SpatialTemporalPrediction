import numpy as np
from tensorflow import keras
from collections import OrderedDict
import contextlib
import matplotlib.pyplot as plt


def mean_squared_error(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    diff = a - b
    return sum(diff ** 2) / len(diff)


def evaluate_single_simulation(sim, nn_model, input_length, lst_of_lengths):
    mses = np.empty((len(lst_of_lengths)))

    sim = np.asarray(sim)
    n = sum(sim[0])
    normalized_sim = sim / n

    max_forecast_length = max(lst_of_lengths)

    future_data = np.empty((0, 4))

    current_data = normalized_sim[:input_length]
    current_data = np.expand_dims(current_data, 0)

    for step in range(max_forecast_length):
        next_data = np.asarray(nn_model.predict_step(current_data))[0]
        current_data = current_data[:, 1:]
        current_data = np.append(current_data, [[next_data]], axis=1)
        future_data = np.append(future_data, [next_data], axis=0)

    for index in range(len(lst_of_lengths)):
        forecast_length = lst_of_lengths[index]
        forecast_index = forecast_length - 1
        prediction = future_data[forecast_index]
        actual_data = normalized_sim[input_length + forecast_index]
        mse = mean_squared_error(prediction, actual_data)
        mses[index] = mse

    return mses


def evaluate(val_sims, nn_model, input_length, seq_of_lengths):
    ret = OrderedDict()
    mses = np.empty((0, len(seq_of_lengths)))
    for sim in val_sims:
        mses_to_add = evaluate_single_simulation(sim, nn_model, input_length, seq_of_lengths)
        mses = np.append(mses, [mses_to_add], axis=0)

    average_mses = np.mean(mses, axis=0)
    stds = np.std(mses, axis=0)

    nn_model.summary()
    ret['Number of validation simulations'] = len(val_sims)
    ret['Length of inputs'] = input_length
    ret['Evaluation distances'] = seq_of_lengths
    ret['Average mean squared error for each distance'] = average_mses
    ret['Standard deviation of mean squared error'] = stds

    [print(key, ':', value) for key, value in ret.items()]
    print('MSE values :')
    print(mses)
    print('\n')


if __name__ == '__main__':
    FILE_PATH = '../../data/data_200_sims.npz'
    TRAIN_NUM = 100
    VAL_NUM = 50
    TEST_NUM = 50
    EVALUATION_DIST = 50

    MODEL_PATH = 'models/m6'
    PRIOR_STEPS = 50
    FUTURE_STEPS = 1

    all_sims = np.load(FILE_PATH)
    val_file_names = all_sims.files[TRAIN_NUM:TRAIN_NUM + VAL_NUM]
    validation_sims = []

    for elem in validation_sims:
        print(len(elem))

    for file_name in val_file_names:
        validation_sims.append(all_sims[file_name])

    model = keras.models.load_model(MODEL_PATH)

    file_path = 'evaluations.txt'
    with open(file_path, "a") as f:
        with contextlib.redirect_stdout(f):
            evaluate(validation_sims, model, PRIOR_STEPS, [1, 5, 20, 50])
