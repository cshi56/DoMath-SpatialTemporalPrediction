import numpy as np
import random
import math
import matplotlib.pyplot as plt
from time import process_time

random.seed(1234)
np.random.seed(1234)
np.set_printoptions(threshold=np.inf)


class Simulation:
    def __init__(self, beta, a, gamma, n, s=0, e=0, i=0):
        """
        Specify initial parameters to initialize simulation.
        """
        self.beta = beta
        self.a = a
        self.gamma = gamma
        self.n = n
        self.s = s
        self.e = e
        self.i = i
        self.r = n - s - e - i
        self.total_out_of_s_coefficient = 0
        self.total_into_i_coefficient = 0
        self.total_into_r_coefficient = 0
        self.history = [[self.s, self.e, self.i, self.r]]
        self.total_time = 0
        self.time_steps = [0]
        self.unit_time_data = [[self.s, self.e, self.i, self.r]]
        self.unit_time_step = 1

    def update_coefficients(self):
        """
        This updates the coefficients/parameters used for each of the three Poisson
        processes.
        """
        self.total_out_of_s_coefficient = self.beta * self.i * self.s / self.n
        self.total_into_i_coefficient = self.a * self.e
        self.total_into_r_coefficient = self.gamma * self.i

    def update_seir(self, argmin):
        """
        Updates SEIR categories based on positive numbers produced from sampling the
        Poisson distributions.
        """
        if argmin == 0:
            self.s = self.s - 1
            self.e = self.e + 1
        elif argmin == 1:
            self.e = self.e - 1
            self.i = self.i + 1
        else:
            self.i = self.i - 1
            self.r = self.r + 1

    def update_history(self):
        """
        self.history is a numpy array that stores the values of S, E, I, and R at
        each time step in a simulation. It is updated at every time step.
        """
        self.history.append([self.s, self.e, self.i, self.r])

    def time_step(self):
        if self.e == 0 and self.i == 0:
            self.history.append([self.s, self.e, self.i, self.r])
            self.total_time += 1
            self.time_steps.append(self.total_time)
            return
        self.update_coefficients()
        """
        This function calculates a random time step for each of the three Poisson processes,
        selects the minimum one, and then updates the categories accordingly. 
        """
        if self.total_out_of_s_coefficient == 0:
            time_for_out_of_s = np.inf
        else:
            time_for_out_of_s = -1 * math.log(random.uniform(0, 1)) / self.total_out_of_s_coefficient

        if self.total_into_i_coefficient == 0:
            time_for_into_i = np.inf
        else:
            time_for_into_i = -1 * math.log(random.uniform(0, 1)) / self.total_into_i_coefficient

        if self.total_into_r_coefficient == 0:
            time_for_into_r = np.inf
        else:
            time_for_into_r = -1 * math.log(random.uniform(0, 1)) / self.total_into_r_coefficient

        time_array = np.asarray([time_for_out_of_s, time_for_into_i, time_for_into_r])
        self.total_time += min(time_array)
        self.time_steps.append(self.total_time)
        argmin = np.argmin(time_array)

        while self.unit_time_step <= self.total_time:
            self.unit_time_data.append([self.s, self.e, self.i, self.r])
            self.unit_time_step += 1

        self.update_seir(argmin)
        self.update_history()

    def simulate(self, total_time):
        """
        Simulates SEIR model given specified number of times steps.
        """
        while self.time_steps[-1] < total_time:
            self.time_step()

        self.unit_time_data.append([self.s, self.e, self.i, self.r])

    def simulate_till_end(self):
        """
        Simulates SEIR model until E and I categories are zero
        """
        while self.e != 0 or self.i != 0:
            self.time_step()

        self.unit_time_data.append([self.s, self.e, self.i, self.r])

    def graph(self):
        """
        Graphs the data stored in self.history.
        """
        time_range = range(self.unit_time_step + 1)

        s_unit_data = np.asarray(self.unit_time_data)[:, 0]
        e_unit_data = np.asarray(self.unit_time_data)[:, 1]
        i_unit_data = np.asarray(self.unit_time_data)[:, 2]
        r_unit_data = np.asarray(self.unit_time_data)[:, 3]

        plt.plot(time_range, s_unit_data, label='Susceptible subjects')
        plt.plot(time_range, e_unit_data, label='Exposed subjects')
        plt.plot(time_range, i_unit_data, label='Infected subjects')
        plt.plot(time_range, r_unit_data, label='Removed subjects')
        plt.legend()
        plt.show()


def graph_beta(a, gamma, n, s, i):
    """
    Shows how changes in beta parameter affect total number of infected cases.
    """
    beta = 0
    results = []
    for beta in np.linspace(0, .25, 200):
        simul = Simulation(beta, a, gamma, n, s=s, i=i)
        simul.simulate_till_end()
        final = simul.history[-1][3]
        results.append(final)
    plt.plot(np.linspace(0, .25, 200), results)
    plt.xlabel('beta value')
    plt.ylabel('Cumulative number of infected at end of simulation')
    plt.show()


if __name__ == '__main__':
    "setting variables"
    beta = .4  # number of contacts per person per time step
    a = .1  # parameter controlling latency between exposure and infection
    gamma = .05  # parameter specifying probability of removal
    n = 500000  # total population
    i = 100  # initial number of infected subjects
    s = n - i  # susceptible subjects

    sim = Simulation(beta, a, gamma, n, s=s, i=i)
    sim.simulate_till_end()
    sim.graph()
    print(np.asarray(sim.unit_time_data))
