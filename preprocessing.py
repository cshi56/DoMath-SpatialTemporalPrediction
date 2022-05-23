import numpy as np

np.set_printoptions(threshold=np.inf)


def make_data(simulation_list, prior_steps):
    ret = []
    for simulation in simulation_list:
        if len(simulation) < prior_steps + 1:
            continue
        for i in range(len(simulation) - prior_steps):
            datum = simulation[i: i + prior_steps + 1]
            ret.append(datum)
    return ret


def preprocess(file, split_ratio, prior_steps):
    all_data = np.load(file)
    file_names = all_data.files
    training_number = int(len(file_names) * split_ratio)
    training_simulations = []
    testing_simulations = []

    for i in range(training_number):
        training_simulations.append(all_data[file_names[i]])
    for i in range(training_number, len(file_names)):
        testing_simulations.append(all_data[file_names[i]])

    training_data = np.asarray(make_data(training_simulations, prior_steps))
    testing_data = np.asarray(make_data(testing_simulations, prior_steps))

    np.random.shuffle(training_data)
    np.random.shuffle(testing_data)

    return training_data, testing_data


if __name__ == '__main__':
    train_data, test_data = preprocess('data.npz', 0.8, 20)
    print(train_data)