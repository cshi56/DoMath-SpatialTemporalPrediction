import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.random.seed(1234)


def zero_one_normalize(sim_data):
    return sim_data / sum(sim_data[0])


def make_data(simulation_list, prior_steps, future_steps=1, stride=1):
    ret = []
    for simulation in simulation_list:
        start = np.random.randint(0, stride)
        if len(simulation) < prior_steps + future_steps + start:
            continue
        for i in range(start, len(simulation) - prior_steps - future_steps + 1, stride):
            datum = simulation[i: i + prior_steps + future_steps]
            ret.append(datum)
    return ret


def preprocess(file, train_num, val_num, test_num, prior_steps, future_steps, stride=1):
    all_data = np.load(file)
    file_names = all_data.files
    num_files = len(file_names)
    if num_files < train_num + val_num + test_num:
        print('Not enough data. File contains ' + str(num_files) + ' simulations, ' +
              str(train_num + val_num + test_num) + ' needed.')
        return 0, 0, 0, 0, 0, 0
    train_sims = []
    val_sims = []
    test_sims = []

    for i in range(train_num):
        sim_data = np.asarray(all_data[file_names[i]])
        sim_data = zero_one_normalize(sim_data)
        train_sims.append(sim_data)
    for i in range(train_num, train_num + val_num):
        sim_data = np.asarray(all_data[file_names[i]])
        sim_data = zero_one_normalize(sim_data)
        val_sims.append(sim_data)
    for i in range(train_num + val_num, train_num + val_num + test_num):
        sim_data = np.asarray(all_data[file_names[i]])
        sim_data = zero_one_normalize(sim_data)
        test_sims.append(sim_data)

    training_data = np.asarray(make_data(train_sims, prior_steps, future_steps=future_steps, stride=stride))
    validation_data = np.asarray(make_data(val_sims, prior_steps, future_steps=future_steps, stride=stride))
    testing_data = np.asarray(make_data(test_sims, prior_steps, future_steps=future_steps, stride=stride))

    x_train = training_data[:, :prior_steps]
    y_train = training_data[:, prior_steps:]
    x_val = validation_data[:, :prior_steps]
    y_val = validation_data[:, prior_steps:]
    x_test = testing_data[:, :prior_steps]
    y_test = testing_data[:, prior_steps:]

    y_train = y_train.reshape(len(x_train), 4 * future_steps)
    y_val = y_val.reshape(len(x_val), 4 * future_steps)
    y_test = y_test.reshape(len(x_test), 4 * future_steps)

    return x_train, y_train, x_val, y_val, x_test, y_test


def noisify(data, noise_amount):
    return np.asarray(np.fmax(data + noise_amount * data * np.random.uniform(-1, 1, data.shape),
                              np.zeros(data.shape)), dtype=int)


if __name__ == '__main__':
    x_training, y_training, x_valid, y_valid, x_testing, y_testing = \
        preprocess('../../data/data_200_sims.npz', 100, 50, 30, 50, 1, 1)
    print(y_training[0])
