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
        r = current_seir[3]
        delta_s = (-1 * beta * i * s / n) * time_increment
        delta_e = ((beta * i * s / n) - (alpha * e)) * time_increment
        delta_i = ((alpha * e) - (gamma * i)) * time_increment
        delta_r = (gamma * i) * time_increment
        current_seir = current_seir + np.asarray([delta_s, delta_e, delta_i, delta_r])
        if step % steps_per_day == 0:
            all_seir.append(current_seir)
    return np.asarray(all_seir)


def c(seir_data, t):
    if t >= len(seir_data):
        print('You picked a value of t that is too big.')
        return
    return seir_data[t][2] + seir_data[t][3] - seir_data[t - 1][2] - seir_data[t - 1][3]


def probability(seir_data, alpha, beta, gamma):
    initial_seir_values = seir_data[0]
    s_init = initial_seir_values[0]
    e_init = initial_seir_values[1]
    i_init = initial_seir_values[2]
    seir_length = len(seir_data)
    n = sum(seir_data[0])
    deterministic_seir_data = seir_from_deterministic_model(n,
                                                            s_init,
                                                            e_init,
                                                            i_init,
                                                            alpha,
                                                            beta,
                                                            gamma,
                                                            seir_length - 1)
    prob = Decimal(1)
    for i in range(1, len(seir_data)):
        c_tilde_i = float(c(deterministic_seir_data, i))
        c_i = float(c(seir_data, i))
        if c_i == 0 and c_tilde_i == 0:
            continue
        log_a = c_i * np.log(c_tilde_i) - c_tilde_i
        log_b = math.lgamma(c_i + 1)
        factor = np.exp(log_a - log_b)
        prob *= Decimal(factor)
    return prob


def neg_probability_func_maker(seir_data):
    def neg_probability(parameters):
        ret = -1 * probability(seir_data, parameters[0], parameters[1], parameters[2])
        return ret
    return neg_probability


def maximize_probability(seir_data, tolerance):
    neg_prob = neg_probability_func_maker(seir_data)

    def con1(x):
        return x[0] - 0.07

    def con2(x):
        return 0.14 - x[0]

    def con3(x):
        return x[1] - 0.1

    def con4(x):
        return 0.5 - x[1]

    def con5(x):
        return x[2] - 0.02

    def con6(x):
        return 0.07 - x[2]

    con_1 = {'type': 'ineq', 'fun': con1}
    con_2 = {'type': 'ineq', 'fun': con2}
    con_3 = {'type': 'ineq', 'fun': con3}
    con_4 = {'type': 'ineq', 'fun': con4}
    con_5 = {'type': 'ineq', 'fun': con5}
    con_6 = {'type': 'ineq', 'fun': con6}

    cons = [con_1, con_2, con_3, con_4, con_5, con_6]

    x0 = np.asarray([0.1, 0.3, 0.05])
    result = optimize.minimize(neg_prob, x0, method='COBYLA',
                               constraints=cons,
                               tol=tolerance,
                               options={'maxiter': 100000})
    print(result)
    return result.x


if __name__ == '__main__':
    N = 500000
    S = 490000
    E = 0
    I = 100000
    R = 0
    ALPHA = 0.1
    BETA = 0.2
    GAMMA = 0.04
    TIME = 20
    TOLERANCE = 1E-100

    data = seir_from_deterministic_model(N, S, E, I, ALPHA, BETA, GAMMA, TIME)

    params = maximize_probability(data, TOLERANCE)
    print(params)
    print(probability(data, params[0], params[1], params[2]))
    print(probability(data, ALPHA, BETA, GAMMA))
