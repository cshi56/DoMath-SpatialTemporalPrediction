from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import optimize
from decimal import *

np.random.seed(1234)


def seir_from_deterministic_model(n, s_zero, e_zero, i_zero, alpha, beta, gamma, time, steps_per_day=10):
    r_zero = n - s_zero - e_zero - i_zero
    current_seir = np.asarray([s_zero, e_zero, i_zero, r_zero], dtype=float)
    all_seir = [current_seir]

    time_increment = 1 / steps_per_day

    for step in range(steps_per_day * time):
        s = current_seir[0]
        e = current_seir[1]
        i = current_seir[2]

        delta_s = (-1 * beta * i * s / n) * time_increment
        delta_e = ((beta * i * s / n) - (alpha * e)) * time_increment
        delta_i = ((alpha * e) - (gamma * i)) * time_increment
        delta_r = -1 * (delta_s + delta_e + delta_i)

        current_seir = current_seir + np.asarray([delta_s, delta_e, delta_i, delta_r])

        if delta_r < 0:
            print(delta_r)
            print(alpha, beta, gamma)

        if step % steps_per_day == 0:
            all_seir.append(current_seir)

    return np.asarray(all_seir)


def c(seir_data, t):
    c_val = seir_data[t][2] + seir_data[t][3] - seir_data[t - 1][2] - seir_data[t - 1][3]
    return c_val


def c_tilde(seir_data, alpha, beta, gamma, t):
    initial_data = seir_data[t - 1]
    s_init = initial_data[0]
    e_init = initial_data[1]
    i_init = initial_data[2]
    n = sum(initial_data)

    next_data = seir_from_deterministic_model(n, s_init, e_init, i_init, alpha, beta, gamma, 1)

    c_val = next_data[1][2] + next_data[1][3] - initial_data[2] - initial_data[3]
    return c_val


#def c_gamma(seir_data, t):
#    return (seir_data[t][3] - seir_data[t - 1][3]) / (seir_data[t - 1][3] + 1)


#def c_gamma_tilde(seir_data, alpha, beta, gamma, t):
#    initial_data = seir_data[t - 1]
#    s_init = initial_data[0]
#    e_init = initial_data[1]
#    i_init = initial_data[2]
#    n = sum(initial_data)
#    next_data = seir_from_deterministic_model(n, s_init, e_init, i_init, alpha, beta, gamma, 1)
#    return (next_data[1][3] - initial_data[3]) / (initial_data[3] + 1)


def probability(seir_data, alpha, beta, gamma):
    log_prob = 0

    for i in range(1, len(seir_data)):
        c_tilde_i = float(c_tilde(seir_data, alpha, beta, gamma, i))
        c_i = float(c(seir_data, i))

        if c_tilde_i == 0:
            continue

        if c_i == 0 and c_tilde_i == 0:
            continue

        log_a = c_i * np.log(c_tilde_i) - c_tilde_i
        log_b = math.lgamma(c_i + 1)

        log_prob += log_a - log_b

    return log_prob


#def gamma_probability(seir_data, gamma):
#    log_prob = 0
#    for i in range(1, len(seir_data)):
#        c_tilde_i = float(c_gamma_tilde(seir_data, 0.07, 0.1, gamma, i))
#        c_i = float(c_gamma(seir_data, i))
#        if c_i == 0 and c_tilde_i == 0:
#            continue
#        log_a = c_i * np.log(c_tilde_i) - c_tilde_i
#        log_b = math.lgamma(c_i + 1)
#        log_prob += log_a - log_b
#    return log_prob


def neg_probability_func_maker(seir_data):
    def neg_probability(parameters):
        ret = -1 * probability(seir_data, parameters[0], parameters[1], parameters[2])
        return ret
    return neg_probability


#def neg_probability_func_maker_gamma(seir_data):
#    def neg_probability(gam):
#        ret = -1 * gamma_probability(seir_data, gam)
#        return ret
#    return neg_probability


def maximize_probability(seir_data, tolerance):
    neg_prob = neg_probability_func_maker(seir_data)

    x0 = np.asarray([0.1, 0.3, 0.05])

    lb = np.asarray([0.04, 0.07, 0.01])
    ub = np.asarray([0.17, 0.6, 0.09])
    bounds = optimize.Bounds(lb, ub)

    result = optimize.minimize(neg_prob, x0, method='nelder-mead',
                               bounds=bounds,
                               tol=tolerance,
                               options={'maxiter': 10000},)
    print(result)
    return result.x


#def maximize_probability_gamma(seir_data, tolerance):
#    neg_prob = neg_probability_func_maker_gamma(seir_data)
#
#    x0 = np.asarray([0.05])
#    lb = np.asarray([0.01])
#    ub = np.asarray([0.09])
#    bounds = optimize.Bounds(lb, ub)
#
#    result = optimize.minimize(neg_prob, x0, method='nelder-mead',
#                               bounds=bounds,
#                               tol=tolerance,
#                               options={'maxiter': 100,
#                               'xatol': 1E-100,
#                               'fatol':1E-100},)
#    print(result)
#    return result.x


if __name__ == '__main__':
    N = 5000
    S = 4900
    E = 0
    I = 100
    R = 0
    ALPHA = 0.1
    BETA = 0.4
    GAMMA = 0.04
    TIME = 50
    TOLERANCE = 1E-100

    data = seir_from_deterministic_model(N, S, E, I, ALPHA, BETA, GAMMA, TIME)

    params = maximize_probability(data, TOLERANCE)

    print(params)
    print(probability(data, params[0], params[1], params[2]))
    print(probability(data, ALPHA, BETA, GAMMA))
