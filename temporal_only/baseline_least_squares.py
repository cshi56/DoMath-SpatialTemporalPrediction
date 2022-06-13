from simulation import Simulation
import numpy as np
from scipy import optimize
from baseline_posterior_prob import seir_from_deterministic_model
import matplotlib.pyplot as plt
from preprocessing import noisify

np.random.seed(1234)


def mse(seir_data1, seir_data2):
    seir_data1 = np.asarray(seir_data1)
    seir_data2 = np.asarray(seir_data2)
    return ((seir_data1 - seir_data2) ** 2).mean()


def mse_from_params(seir_data, alpha, beta, gamma):
    init_data = seir_data[0]
    n = sum(init_data)
    s_init = init_data[0]
    e_init = init_data[1]
    i_init = init_data[2]
    time = len(seir_data) - 1

    seir_det = seir_from_deterministic_model(n, s_init, e_init, i_init, alpha, beta, gamma, time)

    return mse(seir_data, seir_det)


def mse_function_maker(seir_data):
    def mse_function(params):
        return mse_from_params(seir_data, params[0], params[1], params[2])
    return mse_function


def minimize_mse(seir_data, tolerance):
    mse_func = mse_function_maker(seir_data)

    x0 = np.asarray([0.1, 0.3, 0.05])

    lb = np.asarray([0.03, 0.07, 0.01])
    ub = np.asarray([0.18, 0.6, 0.09])
    bounds = optimize.Bounds(lb, ub)

    result = optimize.minimize(mse_func, x0, method='nelder-mead',
                               bounds=bounds,
                               tol=tolerance,
                               options={'maxiter': 1000,
                                        'xatol': tolerance,
                                        'fatol': tolerance}, )
    return result


if __name__ == '__main__':
    N = 500000
    E = 0
    I = 10000
    R = 0
    S = N - I - E - R
    ALPHA = 0.14
    BETA = 0.5
    GAMMA = 0.07
    TIME = 200
    TOLERANCE = 1E-100

    INPUT_LENGTH = 50
    LENGTH_OF_FORECAST = 50
    STRIDE = 50

    data = np.load('../../data/data_200_sims.npz')
    file_names = data.files
    val_sim_names = file_names[100:150]
    val_sims = []
    for file_name in val_sim_names:
        seir_info = data[file_name]
        noised = seir_info
        val_sims.append(noised)

    mses = []
    total_mse = 0
    mse_count = 0
    simnum = 1

    for seir_data in val_sims:
        print('sim num:', simnum)
        simnum += 1
        n = sum(seir_data[0])
        seir_data = seir_data / n
        sim_length = len(seir_data)

        for t in range(0, sim_length - INPUT_LENGTH - LENGTH_OF_FORECAST, STRIDE):
            initial_data = seir_data[t:t + INPUT_LENGTH]
            estimated_params = minimize_mse(initial_data, TOLERANCE).x
            alpha = estimated_params[0]
            beta = estimated_params[1]
            gamma = estimated_params[2]
            n = sum(initial_data[-1])
            s = initial_data[-1][0]
            e = initial_data[-1][1]
            i = initial_data[-1][2]
            time = LENGTH_OF_FORECAST

            predicted_data = seir_from_deterministic_model(n, s, e, i, alpha, beta, gamma, time)[1:]
            actual_data = seir_data[t + INPUT_LENGTH:t + INPUT_LENGTH + LENGTH_OF_FORECAST]
            total_mse += mse(predicted_data, actual_data)
            mses.append(mse(predicted_data, actual_data))
            mse_count += 1
            print('av mse so far', str(total_mse / mse_count))

    print('average overall mse: ', str(total_mse / mse_count))
    mses = np.asarray(mses)
    print('mean: ', str(mses.mean()))
    print('std: ', str(mses.std()))
    print(mses)

