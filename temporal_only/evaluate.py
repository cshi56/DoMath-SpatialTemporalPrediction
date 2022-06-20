import numpy as np
from tensorflow import keras
from collections import OrderedDict
import contextlib


def reshape_by_output(vector):
    if len(vector) == 4:
        return vector
    return np.reshape(vector, (int(len(vector) / 4), 4))


def mean_squared_error(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    diff = a - b
    return sum(diff ** 2) / len(diff)


def predict_future(sim, nn_model, input_length, length):
    sim = np.asarray(sim)
    n = sum(sim[0])
    normalized_sim = sim / n

    future_data = np.empty((0, 4))

    current_data = normalized_sim[:input_length]
    current_data = np.expand_dims(current_data, 0)

    while len(future_data) < length:
        next_data = np.asarray(nn_model.predict_step(current_data))[0]
        next_data = reshape_by_output(next_data)

        future_steps = int(len(next_data.reshape(-1)) / 4)

        current_data = current_data[:, future_steps:]
        current_data = reshape_by_output(np.append(current_data.reshape(-1), next_data.reshape(-1)))
        current_data = np.expand_dims(current_data, axis=0)
        future_data = reshape_by_output(np.append(future_data.reshape(-1), next_data.reshape(-1)))

    future_data = future_data * n

    return future_data[:length]


def evaluate_single_simulation(sim, nn_model, input_length, length):
    sim = np.asarray(sim)
    n = sum(sim[0])
    normalized_sim = sim / n

    future_data = np.empty((0, 4))

    current_data = normalized_sim[:input_length]
    current_data = np.expand_dims(current_data, 0)

    while len(future_data) < length:
        next_data = np.asarray(nn_model.predict_step(current_data))[0]
        next_data = reshape_by_output(next_data)

        future_steps = int(len(next_data.reshape(-1)) / 4)

        current_data = current_data[:, future_steps:]
        current_data = reshape_by_output(np.append(current_data.reshape(-1), next_data.reshape(-1)))
        current_data = np.expand_dims(current_data, axis=0)
        future_data = reshape_by_output(np.append(future_data.reshape(-1), next_data.reshape(-1)))

    forecast_index = length - 1
    predicted_i = future_data[forecast_index][2]
    actual_i = normalized_sim[input_length + forecast_index][2]

    return actual_i, predicted_i


def evaluate(val_sims, nn_model, input_length, length):
    ret = OrderedDict()
    actual_predicted_is = np.empty((0, 2))
    for sim in val_sims:
        actual_i, predicted_i = evaluate_single_simulation(sim, nn_model, input_length, length)
        actual_predicted_is = np.append(actual_predicted_is, [[actual_i, predicted_i]], axis=0)

    aes = np.abs(actual_predicted_is[:, 0] - actual_predicted_is[:, 1])
    actual_is = actual_predicted_is[:, 0]
    wae = np.sum(aes) / np.sum(actual_is)
    wape = wae * 100

    nn_model.summary()
    ret['Number of validation simulations'] = len(val_sims)
    ret['Length of inputs'] = input_length
    ret['Evaluation distance'] = length
    ret['Weighted absolute percentage error of I at future time step ' + str(length)] = "{:.2f}".format(wape) + '%'

    [print(key, ':', value) for key, value in ret.items()]
    print('Actual and predicted values of I at future time step :')
    print(actual_predicted_is)
    print('\n')


if __name__ == '__main__':
    FILE_PATH = '../../data/data_200_sims.npz'
    TRAIN_NUM = 100
    VAL_NUM = 50
    TEST_NUM = 50
    EVALUATION_DIST = 50

    PRIOR_STEPS = 50

    all_sims = np.load(FILE_PATH)
    val_file_names = all_sims.files[TRAIN_NUM:TRAIN_NUM + VAL_NUM]
    validation_sims = []

    for elem in validation_sims:
        print(len(elem))

    for file_name in val_file_names:
        validation_sims.append(all_sims[file_name])

    file_path = 'evaluations2.txt'
    with open(file_path, "a") as f:
        with contextlib.redirect_stdout(f):
            for i in range(13, 14):
                model_path = 'models/m' + str(i)
                model = keras.models.load_model(model_path, compile=False)
                evaluate(validation_sims, model, PRIOR_STEPS, EVALUATION_DIST)
