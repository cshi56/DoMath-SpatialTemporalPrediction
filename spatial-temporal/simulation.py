import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1234)
np.random.seed(1234)
np.set_printoptions(threshold=np.inf)


def my_log(number):
    if number == 0:
        return -1 * np.infty
    return np.log(number)


def my_divide(a, b):
    if b == 0:
        return np.sign(a) * np.infty
    return a / b


class Node:
    def __init__(self, alpha, beta, gamma, n, s, e, i):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n = n

        self.s = s
        self.e = e
        self.i = i
        self.r = n - s - e - i

        self.s_to_e_coefficient = beta * i * s / n
        self.e_to_i_coefficient = alpha * e
        self.i_to_r_coefficient = gamma * i

        self.total_time = 0.0
        self.history = [[self.total_time, self.s, self.e, self.i, self.r]]

        self.next_unit_time_step = 1
        self.unit_time_history = [[0, self.s, self.e, self.i, self.r]]

    def update_coefficients(self):
        self.s_to_e_coefficient = self.beta * self.i * self.s / self.n
        self.e_to_i_coefficient = self.alpha * self.e
        self.i_to_r_coefficient = self.gamma * self.i

    def update_seir(self, argmin):
        if argmin == 0:
            self.s -= 1
            self.e += 1
        elif argmin == 1:
            self.e -= 1
            self.i += 1
        else:
            self.i -= 1
            self.r += 1

    def time_step(self):
        self.update_coefficients()

        if max(self.s_to_e_coefficient, self.e_to_i_coefficient, self.i_to_r_coefficient) == 0:
            self.total_time += 0.1
            self.history.append([self.total_time, self.s, self.e, self.i, self.r])
            return

        s_to_e_time = -1 * my_divide(my_log(random.uniform(0, 1)), self.s_to_e_coefficient)
        e_to_i_time = -1 * my_divide(my_log(random.uniform(0, 1)), self.e_to_i_coefficient)
        i_to_r_time = -1 * my_divide(my_log(random.uniform(0, 1)), self.i_to_r_coefficient)

        time_array = np.asarray([s_to_e_time, e_to_i_time, i_to_r_time])
        self.total_time += min(time_array)
        argmin = np.argmin(time_array)
        self.update_seir(argmin)
        self.history.append([self.total_time, self.s, self.e, self.i, self.r])

    def graph(self):
        history = np.asarray(self.unit_time_history)

        time_data = history[:, 0]
        s_data = history[:, 1]
        e_data = history[:, 2]
        i_data = history[:, 3]
        r_data = history[:, 4]

        plt.plot(time_data, s_data, label='Susceptible subjects')
        plt.plot(time_data, e_data, label='Exposed subjects')
        plt.plot(time_data, i_data, label='Infected subjects')
        plt.plot(time_data, r_data, label='Removed subjects')
        plt.legend()
        plt.show()

    def simulate_single_time_unit(self):
        while self.total_time <= self.next_unit_time_step:
            self.time_step()

        self.history.pop()
        self.total_time = self.next_unit_time_step

        unit_data_to_add = [self.next_unit_time_step]
        unit_data_to_add.extend(self.history[-1][1:])

        self.unit_time_history.append(unit_data_to_add)
        self.history.append(unit_data_to_add)  # can do this due to lack of memory of Poisson processes
        self.s, self.e, self.i, self.r = \
            self.history[-1][1], self.history[-1][2], self.history[-1][3], self.history[-1][4]
        self.next_unit_time_step += 1

    def simulate(self):
        while self.unit_time_history[-1][2] + self.unit_time_history[-1][3] != 0:
            self.simulate_single_time_unit()


class Simulation:
    def __init__(self):
        self.nodes = []
        self.number_of_nodes = 0
        self.diffusion_matrix = np.empty((0, 0))
        self.current_time_step = 1

    def add_node(self, node):
        self.nodes.append(node)
        self.number_of_nodes += 1
        padding = ((0, 1), (0, 1))
        self.diffusion_matrix = np.pad(self.diffusion_matrix,
                                       pad_width=padding,
                                       mode='constant',
                                       constant_values=0.0)

    def populate_diffusion_matrix(self):
        for node_index in range(self.number_of_nodes):
            for index in range(self.number_of_nodes):
                minimal_n = min(self.nodes[node_index].n, self.nodes[index].n)
                if index == node_index:
                    continue
                diffusion_number = int(1 * random.uniform(0, 1) * minimal_n /
                                       (self.number_of_nodes * 4 ** (abs(node_index - index))))
                self.diffusion_matrix[node_index][index] = diffusion_number
                self.diffusion_matrix[index][node_index] = diffusion_number

    def get_accumulated_seir_from_diff_matrix(self, node_index):
        accumulated_seir = [0, 0, 0, 0]

        for index in range(self.number_of_nodes):
            total_diff_number = self.diffusion_matrix[node_index][index]
            node = self.nodes[index]

            s_transfer_in = int(total_diff_number * node.s / node.n)
            e_transfer_in = int(total_diff_number * node.e / node.n)
            i_transfer_in = int(total_diff_number * node.i / node.n)
            r_transfer_in = total_diff_number - s_transfer_in - e_transfer_in - i_transfer_in

            accumulated_seir[0] += s_transfer_in
            accumulated_seir[1] += e_transfer_in
            accumulated_seir[2] += i_transfer_in
            accumulated_seir[3] += r_transfer_in

        for index in range(self.number_of_nodes):
            total_diff_number = self.diffusion_matrix[node_index][index]
            node = self.nodes[node_index]

            s_transfer_out = int(total_diff_number * node.s / node.n)
            e_transfer_out = int(total_diff_number * node.e / node.n)
            i_transfer_out = int(total_diff_number * node.i / node.n)
            r_transfer_out = total_diff_number - s_transfer_out - e_transfer_out - i_transfer_out

            accumulated_seir[0] -= s_transfer_out
            accumulated_seir[1] -= e_transfer_out
            accumulated_seir[2] -= i_transfer_out
            accumulated_seir[3] -= r_transfer_out

        return accumulated_seir

    def diffuse(self):
        accumulated_seir_dic = {}

        for node_index in range(self.number_of_nodes):
            accumulated_seir_dic[node_index] = self.get_accumulated_seir_from_diff_matrix(node_index)

        for node_index in range(self.number_of_nodes):
            self.nodes[node_index].s += accumulated_seir_dic[node_index][0]
            self.nodes[node_index].e += accumulated_seir_dic[node_index][1]
            self.nodes[node_index].i += accumulated_seir_dic[node_index][2]
            self.nodes[node_index].r += accumulated_seir_dic[node_index][3]

    def simulate_single_time_unit(self):
        print('Simulating time step ' + str(self.current_time_step) + '.')
        for node in self.nodes:
            node.simulate_single_time_unit()
        self.diffuse()
        self.current_time_step += 1


if __name__ == '__main__':
    ALPHA = 0.1
    BETA = 0.4
    GAMMA = 0.05
    N = 500000
    S = 499900
    E = 0
    I = 100
    NODES = 100
    STEPS = 2000

    new_sim = Simulation()
    for _ in range(NODES):
        alpha = 0.1
        beta = random.uniform(0.1, 0.5)
        gamma = 0.05
        n = random.randrange(10000, 100000)
        if _ == 0:
            i = 10
        else:
            i = 0
        s = n - i
        e = 0
        node_to_add = Node(alpha, beta, gamma, n, s, 0, i)
        new_sim.add_node(node_to_add)

    new_sim.populate_diffusion_matrix()
    print(new_sim.diffusion_matrix)

    for _ in range(STEPS):
        new_sim.simulate_single_time_unit()

    all_data = np.asarray(new_sim.nodes[0].unit_time_history)
    for index in range(1, NODES):
        this_node = new_sim.nodes[index]
        all_data += np.asarray(this_node.unit_time_history)

    history = all_data

    time_data = history[:, 0] / NODES
    s_data = history[:, 1]
    e_data = history[:, 2]
    i_data = history[:, 3]
    r_data = history[:, 4]

    plt.plot(time_data, e_data, label='Exposed subjects')
    plt.plot(time_data, i_data, label='Infected subjects')
    plt.legend()
    plt.show()
