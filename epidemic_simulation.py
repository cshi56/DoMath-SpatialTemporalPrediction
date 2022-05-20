import numpy as np
import matplotlib.pyplot as plt

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
        self.history = np.asarray([[self.s, self.e, self.i, self.r]])
        self.total_time = 0
        self.time_steps = np.array([0])

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
        self.history = np.append(self.history, [[self.s, self.e, self.i, self.r]], axis=0)

    def time_step(self):
        if self.e == 0 and self.i == 0:
            self.history = np.append(self.history, [[self.s, self.e, self.i, self.r]], axis=0)
            self.total_time += 1
            self.time_steps = np.append(self.time_steps, self.total_time)
            return
        self.update_coefficients()
        """
        This function calculates a random time step for each of the three Poisson processes,
        selects the minimum one, and then updates the categories accordingly. 
        """
        if self.total_out_of_s_coefficient == 0:
            time_for_out_of_s = np.inf
        else:
            time_for_out_of_s = -1 * np.log(np.random.uniform()) / self.total_out_of_s_coefficient

        if self.total_into_i_coefficient == 0:
            time_for_into_i = np.inf
        else:
            time_for_into_i = -1 * np.log(np.random.uniform()) / self.total_into_i_coefficient

        if self.total_into_r_coefficient == 0:
            time_for_into_r = np.inf
        else:
            time_for_into_r = -1 * np.log(np.random.uniform()) / self.total_into_r_coefficient

        time_array = np.asarray([time_for_out_of_s, time_for_into_i, time_for_into_r])
        self.total_time += min(time_array)
        self.time_steps = np.append(self.time_steps, self.total_time)
        argmin = np.argmin(time_array)

        self.update_seir(argmin)
        self.update_history()

    def simulate(self, total_time):
        """
        Simulates SEIR model given specified number of times steps.
        """
        while self.time_steps[-1] < total_time:
            self.time_step()

    def simulate_till_end(self):
        """
        Simulates SEIR model until E and I categories are zero
        """
        while self.e != 0 or self.i != 0:
            self.time_step()

    def graph(self):
        """
        Graphs the data stored in self.history.
        """
        s_data, e_data, i_data, r_data = self.history[:, 0], self.history[:, 1], \
                                         self.history[:, 2], self.history[:, 3]
        plt.plot(self.time_steps, s_data, label='Susceptible subjects')
        plt.plot(self.time_steps, e_data, label='Exposed subjects')
        plt.plot(self.time_steps, i_data, label='Infected subjects')
        plt.plot(self.time_steps, r_data, label='Removed subjects')
        plt.legend()
        plt.show()

    def return_unit_time_data(self):
        """
        Returns SEIR data indexed by unit time steps
        """
        total_time = int(self.time_steps[-1])
        data = np.asarray([self.history[0]])
        for i in range(1, total_time + 1):
            for j in range(len(self.time_steps)):
                if self.time_steps[j] == i:
                    data = np.append(data, [self.history[j]], axis=0)
                    break
                elif self.time_steps[j] > i:
                    data = np.append(data, [self.history[j - 1]], axis=0)
                    break
        if total_time - self.time_steps[-1] != 0:
            data = np.append(data, [self.history[-1]], axis=0)
        return data



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
    beta = .5  # number of contacts per person per time step
    a = .2  # parameter controlling latency between exposure and infection
    gamma = .07  # parameter specifying probability of removal
    n = 1000  # total population
    i = 10  # initial number of infected subjects
    s = n - i  # susceptible subjects

    sim = Simulation(beta, a, gamma, n, s=s, i=i)
    sim.simulate_till_end()
    sim.graph()