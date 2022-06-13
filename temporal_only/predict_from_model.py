import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from tensorflow import keras
from preprocessing import preprocess, noisify


def predict(simulation_data, nn_model, steps, start):
    """
    Steps describes how many prior steps the network was trained on.
    Start gives the time step of the first out of the 50 initial values
    that the model will predict on.
    nn_model is the model to do the prediction
    """
    normalized_data = np.asarray(simulation_data) / 500000
    initial_data = np.asarray(normalized_data[start:start + steps])
    print(initial_data)
    initial_data = np.expand_dims(initial_data, 0)
    predicted_data = [[0, 0, 0, 0]]
    length_of_sim = len(normalized_data)
    for i in range(length_of_sim - steps - start):
        next_data = np.asarray(nn_model.predict_step(initial_data))[0]
        predicted_data = np.append(predicted_data, [next_data], axis=0)
        initial_data = initial_data[:, 1:]
        initial_data = np.append(initial_data, [[next_data]], axis=1)
    predicted_data = predicted_data[1:] * 500000
    plt.plot(range(length_of_sim), simulation_data, label=['actual S', 'actual E', 'actual I', 'actual R'])
    plt.plot(range(start + steps, length_of_sim), predicted_data,
             label=['predicted S', 'predicted E', 'predicted I', 'predicted R'], ls='dotted', lw=2.5)
    plt.legend()
    plt.show()


def evaluation_metric_for_input(validation_data, nn_model, input_length, length_of_forecast, stride):
    total_mse = 0
    mse_count = 0
    sim_num = 1
    for seir_data in validation_data:
        print('sim ' + str(sim_num) + '/50')
        n = sum(seir_data[0])
        seir_data = seir_data / n
        sim_length = len(seir_data)
        for t in range(0, sim_length - input_length - length_of_forecast, stride):
            y_actual = seir_data[t + input_length + length_of_forecast - 1]
            initial_data = seir_data[t:t + input_length]
            initial_data = np.expand_dims(initial_data, 0)
            for _ in range(length_of_forecast):
                next_data = np.asarray(nn_model.predict(initial_data))[0]
                initial_data = initial_data[:, 1:]
                initial_data = np.append(initial_data, [[next_data]], axis=1)
            y_hat = initial_data[0][-1]
            diff = y_actual - y_hat
            mse = sum(diff ** 2) / len(diff)
            total_mse += mse
            mse_count += 1
        sim_num += 1
    return total_mse / mse_count


if __name__ == '__main__':
    LENGTH_OF_FORECAST = 50
    STRIDE = 50

    model = keras.models.load_model('models/m5')

    data = np.load('../../data/data_200_sims.npz')
    file_names = data.files
    validation_sims = []
    for i in range(100, 150):
        validation_sims.append(np.asarray(data[file_names[i]]))

    #print(evaluation_metric_for_input(validation_sims, model, 50, LENGTH_OF_FORECAST, STRIDE))

    for sim_data in validation_sims:
        #  sim_data = noisify(sim_data, 0.15)
        predict(sim_data, model, 50, 0)
