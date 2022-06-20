import matplotlib.pyplot as plt
from evaluate import predict_future
from baseline_posterior_prob import seir_from_deterministic_model
from baseline_least_squares import minimize_mse
import numpy as np
import tensorflow as tf

FILE = 'data_single_sim_100_times.npy'
MODEL = 6
FUTURE_STEPS = 130
COLORS = ['pink', 'red', 'orange', 'green', 'blue', 'purple']

complexities = [17924, 68612, 76612, 93124, 121924, 134340]
complexity_wapes = [25.77, 23.24, 21.23, 12.40, 14.85, 11.97]

future_steps = [1, 2, 5, 10]
future_wapes = [11.97, 13.86, 16.78, 11.14]

training_sizes = [2845, 5338, 10667, 21340]
training_wapes = [11.97, 6.79, 7.90, 5.32]

plt.plot(complexities, complexity_wapes)
plt.axhline(y=1.81, ls='dotted', c='red', label='Baseline')
plt.scatter(complexities, complexity_wapes)
plt.annotate('Model 1', (complexities[0] - 6000, complexity_wapes[0] + 1))
plt.annotate('Model 2', (complexities[1] - 19000, complexity_wapes[1] - 1.25))
plt.annotate('Model 3', (complexities[2] + 3500, complexity_wapes[2] - 0.5))
plt.annotate('Model 4', (complexities[3] - 7000, complexity_wapes[3] - 1.5))
plt.annotate('Model 5', (complexities[4] - 9000, complexity_wapes[4] + 0.75))
plt.annotate('Model 6', (complexities[5] - 8000, complexity_wapes[5] - 2))
plt.xlim([0, 150000])
plt.ylim([0, 30])
plt.xlabel('Total trainable parameters')
plt.ylabel('WAPE of I at future time step 50')
plt.title('Model Complexity vs Performance')
plt.legend()
plt.show()

plt.plot(future_steps, future_wapes)
plt.axhline(y=1.81, ls='dotted', c='red', label='Baseline')
plt.scatter(future_steps, future_wapes)
plt.annotate('Model 6', (future_steps[0] - 0.6, future_wapes[0] - 1.2))
plt.annotate('Model 7', (future_steps[1] - 0.8, future_wapes[1] + 0.6))
plt.annotate('Model 8', (future_steps[2] - 0.5, future_wapes[2] + 0.55))
plt.annotate('Model 9', (future_steps[3] - 0.5, future_wapes[3] - 1.15))
plt.xlim([0, 11])
plt.ylim([0, 20])
plt.xlabel('Future steps generated per prediction')
plt.ylabel('WAPE of I at future time step 50')
plt.title('Prediction Length vs Performance')
plt.legend()
plt.show()

plt.plot(training_sizes, training_wapes)
plt.axhline(y=1.81, ls='dotted', c='red', label='Baseline')
plt.scatter(training_sizes, training_wapes)
plt.annotate('Model 6', (training_sizes[0] + 500, training_wapes[0]))
plt.annotate('Model 10', (training_sizes[1] - 1450, training_wapes[1] - 1.1))
plt.annotate('Model 11', (training_sizes[2] - 1200, training_wapes[2] + 0.5))
plt.annotate('Model 12', (training_sizes[3] - 1300, training_wapes[3] + 0.5))
plt.xlim([0, 24000])
plt.ylim([0, 15])
plt.xlabel('Size of training set')
plt.ylabel('WAPE of I at future time step 50')
plt.title('Training set size vs Performance')
plt.legend()
plt.show()

stoch_data = np.load(FILE)
for elem in stoch_data:
    plt.plot(elem[:180, 2], color='yellow')

data = np.load(FILE)[0]
seir_data = data[:50 + FUTURE_STEPS]

i_data = seir_data[:, 2]
plt.plot(i_data, label='Ground truth', color='black')
plt.plot(0, color='yellow', label='Range', alpha=1)
plt.axvline(x=50, ls='dotted', c=(0.5, 0.5, 0.5))

data_to_predict_on = seir_data[:50]
n = sum(seir_data[-1])
s = data_to_predict_on[-1][0]
e = data_to_predict_on[-1][1]
i = data_to_predict_on[-1][2]
[alpha, beta, gamma] = minimize_mse(data_to_predict_on, 10E-100).x
rest_of_sim = seir_from_deterministic_model(n, s, e, i, alpha, beta, gamma, 150)[1:131]
plt.plot(range(50, 50 + FUTURE_STEPS), rest_of_sim[:, 2], label='Baseline', ls='dotted', color='darkred', lw=1.5)

for i in range(6):
    color = COLORS[i]
    MODEL = i + 1
    model = tf.keras.models.load_model('models/m' + str(MODEL), compile=False)
    future_data = predict_future(seir_data, model, 50, FUTURE_STEPS)

    future_predicted_i = future_data[:, 2]

    plt.plot(range(50, 50 + FUTURE_STEPS), future_predicted_i, label='Model ' + str(MODEL), c=color)

plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Number of infected')
plt.title('Models 1-6 on Example Simulation')
plt.show()

stoch_data = np.load(FILE)
for elem in stoch_data:
    plt.plot(elem[:180, 2], color='yellow')

data = np.load(FILE)[0]
seir_data = data[:50 + FUTURE_STEPS]

i_data = seir_data[:, 2]
plt.plot(i_data, label='Ground truth', color='black')
plt.plot(0, color='yellow', label='Range', alpha=1)
plt.axvline(x=50, ls='dotted', c=(0.5, 0.5, 0.5))

data_to_predict_on = seir_data[:50]
n = sum(seir_data[-1])
s = data_to_predict_on[-1][0]
e = data_to_predict_on[-1][1]
i = data_to_predict_on[-1][2]
[alpha, beta, gamma] = minimize_mse(data_to_predict_on, 10E-100).x
rest_of_sim = seir_from_deterministic_model(n, s, e, i, alpha, beta, gamma, 150)[1:131]
plt.plot(range(50, 50 + FUTURE_STEPS), rest_of_sim[:, 2], label='Baseline', ls='dotted', color='darkred', lw=1.5)

for i in range(4):
    color = COLORS[i + 1]
    MODEL = i + 6
    model = tf.keras.models.load_model('models/m' + str(MODEL), compile=False)
    future_data = predict_future(seir_data, model, 50, FUTURE_STEPS)

    future_predicted_i = future_data[:, 2]

    plt.plot(range(50, 50 + FUTURE_STEPS), future_predicted_i, label='Model ' + str(MODEL), c=color)

plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Number of infected')
plt.title('Models 6-9 on Example Simulation')
plt.show()

stoch_data = np.load(FILE)
for elem in stoch_data:
    plt.plot(elem[:180, 2], color='yellow')

data = np.load(FILE)[0]
seir_data = data[:50 + FUTURE_STEPS]

i_data = seir_data[:, 2]
plt.plot(i_data, label='Ground truth', color='black')
plt.plot(0, color='yellow', label='Range', alpha=1)
plt.axvline(x=50, ls='dotted', c=(0.5, 0.5, 0.5))

data_to_predict_on = seir_data[:50]
n = sum(seir_data[-1])
s = data_to_predict_on[-1][0]
e = data_to_predict_on[-1][1]
i = data_to_predict_on[-1][2]
[alpha, beta, gamma] = minimize_mse(data_to_predict_on, 10E-100).x
rest_of_sim = seir_from_deterministic_model(n, s, e, i, alpha, beta, gamma, 150)[1:131]
plt.plot(range(50, 50 + FUTURE_STEPS), rest_of_sim[:, 2], label='Baseline', ls='dotted', color='darkred', lw=1.5)

j = 0

for i in [6, 10, 11, 12]:
    color = COLORS[j + 1]
    MODEL = i
    model = tf.keras.models.load_model('models/m' + str(MODEL), compile=False)
    future_data = predict_future(seir_data, model, 50, FUTURE_STEPS)

    future_predicted_i = future_data[:, 2]

    plt.plot(range(50, 50 + FUTURE_STEPS), future_predicted_i, label='Model ' + str(MODEL), c=color)
    j += 1

plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Number of infected')
plt.title('Models 6, 10-12 on Example Simulation')
plt.show()
