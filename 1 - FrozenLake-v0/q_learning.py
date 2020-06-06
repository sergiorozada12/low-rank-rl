import json
import numpy as np
from utils import QLearning, Saver, TestUtils, get_env

parameters_file = "experiments/exp_q_learning.json"
env = get_env()
saver = Saver()
test_utils = TestUtils()
Q_optimal = saver.load_from_pickle("results/Q_optimal.pickle")

with open(parameters_file) as j:
    parameters = json.loads(j.read())

medians = {}
standard_devs = {}
frob_errors = {}

for epsilon in parameters["epsilons"]:

    medians_temp = []
    standard_devs_temp = []
    frob_errors_temp = []

    for i in range(parameters["n_simulations"]):
        q_learner = QLearning(env=env,
                              episodes=parameters["episodes"],
                              max_steps=parameters["max_steps"],
                              epsilon=epsilon,
                              alpha=parameters["alpha"],
                              gamma=parameters["gamma"])

        q_learner.train(check_optimality=True, reference=Q_optimal)

        median, standard_dev = test_utils.smooth_signal(signal=q_learner.steps, w=100)

        medians_temp.append(median)
        standard_devs_temp.append(standard_dev)
        frob_errors_temp.append(q_learner.frobenius_error)

    medians[str(epsilon)] = np.median(medians_temp, axis=0)
    standard_devs[str(epsilon)] = np.median(standard_devs_temp, axis=0)
    frob_errors[str(epsilon)] = np.median(frob_errors_temp, axis=0)

saver.save_to_pickle("results/q_learning_medians.pickle", medians)
saver.save_to_pickle("results/q_learning_stds.pickle", standard_devs)
saver.save_to_pickle("results/q_learning_frob_errors.pickle", frob_errors)