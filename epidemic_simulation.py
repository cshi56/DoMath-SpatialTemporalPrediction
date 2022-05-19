import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)


class Simulation:
    def __init__(self, beta, a, gamma, n, s=0, e=0, i=0):
        self.beta = beta
        self.a = a
        self.gamma = gamma
        self.n = n
        self.s = s
        self.e = e
        self.i = i
        self.r = n - s - e - i
        self.total_out_of_s_coefficient = None
        self.total_into_i_coefficient = None
        self.total_into_r_coefficient = None
        self.history = np.asarray([[self.s, self.e, self.i, self.r]])

    def update_coefficients(self):
        self.total_out_of_s_coefficient = self.beta * self.i * self.s / self.n
        self.total_into_i_coefficient = self.a * self.e
        self.total_into_r_coefficient = self.gamma * self.i

    def update_seir(self, out_of_s, into_i, into_r):
        self.s = self.s - out_of_s
        self.e = self.e + out_of_s - into_i
        self.i = self.i + into_i - into_r
        self.r = self.r + into_r

    def update_history(self):
        self.history = np.append(self.history, [[self.s, self.e, self.i, self.r]], axis=0)

    def time_step(self):
        self.update_coefficients()
        out_of_s = min(self.s, np.random.poisson(self.total_out_of_s_coefficient))
        into_i = min(self.e, np.random.poisson(self.total_into_i_coefficient))
        into_r = min(self.i, np.random.poisson(self.total_into_r_coefficient))

        self.update_seir(out_of_s, into_i, into_r)
        self.update_history()

    def simulate(self, time_steps):
        for _ in range(time_steps):
            self.time_step()

    def graph(self):
        s_data, e_data, i_data, r_data = self.history[:, 0], self.history[:, 1], \
                                         self.history[:, 2], self.history[:, 3]
        plt.plot(s_data, label='Susceptible subjects')
        plt.plot(e_data, label='Exposed subjects')
        plt.plot(i_data, label='Infected subjects')
        plt.plot(r_data, label='Removed subjects')
        plt.legend()
        plt.show()


sim = Simulation(.5, .4, .2, 5000, s=4900, i=100)
sim.simulate(100)
print(sim.history)
sim.graph()








