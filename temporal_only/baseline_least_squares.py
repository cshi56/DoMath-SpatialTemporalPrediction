from simulation import Simulation
import numpy as np
from scipy import optimize
from baseline import seir_from_deterministic_model
import matplotlib.pyplot as plt

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
    TOLERANCE = 1E-10

    sim = Simulation(BETA, ALPHA, GAMMA, N, s=S, e=E, i=I)
    sim.simulate(TIME)
    sim_data = sim.unit_time_data
    print(sim_data)

    data = seir_from_deterministic_model(N, S, E, I, ALPHA, BETA, GAMMA, TIME)

    res = minimize_mse(sim_data[:20], TOLERANCE)

    sim_data_est = seir_from_deterministic_model(N, S, E, I, res.x[0], res.x[1], res.x[2], TIME)

    plt.plot(sim_data)
    plt.plot(sim_data_est)
    plt.show()

    print(res.x)

