import numpy as np
import matplotlib.pyplot as plt
from epidemic_simulation import Simulation
from tensorflow import keras


def predict(simulation_data, nn_model, steps, start):
    """
    Steps describes how many prior steps the network was trained on.
    Start gives the time step of the first out of the 50 initial values
    that the model will predict on.
    nn_model is the model to do the prediction
    """
    normalized_data = np.asarray(simulation_data) / 125000
    initial_data = np.asarray(normalized_data[start:start + steps])
    initial_data = np.expand_dims(initial_data, 0)
    predicted_data = [[0, 0, 0, 0]]
    length_of_sim = len(normalized_data)
    for i in range(length_of_sim - steps - start):
        next_data = np.asarray(nn_model.predict_step(initial_data))[0]
        predicted_data = np.append(predicted_data, [next_data], axis=0)
        initial_data = initial_data[:, 1:]
        initial_data = np.append(initial_data, [[next_data]], axis=1)
    predicted_data = predicted_data[1:] * 125000
    plt.plot(range(length_of_sim), simulation_data, label=['actual S', 'actual E', 'actual I', 'actual R'])
    plt.plot(range(start + steps, length_of_sim), predicted_data,
             label=['predicted S', 'predicted E', 'predicted I', 'predicted R'], ls='dotted', lw=2.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = np.load('data.npz')
    file_names = data.files

    for _ in range(10):
        random_file = file_names[np.random.randint(0, len(file_names))]
        sim_data = data[random_file]
        model = keras.models.load_model('module_1_models/model_51_from_prior_50')
        predict(sim_data, model, 50, 0)
