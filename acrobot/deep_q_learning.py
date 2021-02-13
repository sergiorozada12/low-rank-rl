import json
import numpy as np
from utils import Dqn, Buffer, Saver

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

parameters_file = "experiments/exp_dqn_learning.json"
with open(parameters_file) as j:
    parameters = json.loads(j.read())

env = gym.make('Acrobot-v1')
env._max_episode_steps = np.inf
saver = Saver()

rewards = []
steps = []

for _ in range(parameters["n_simulations"]):

    keras.backend.clear_session()

    alpha = 0.001

    model = Sequential()
    model.add(Dense(parameters["hidden_size"], input_dim=env.observation_space.shape[0], activation='tanh'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=alpha))

    dqn_learner = Dqn(model=model,
                      buffer=Buffer(parameters["buffer_size"]),
                      env=env,
                      gamma=parameters["gamma"],
                      epsilon=parameters["epsilon"],
                      decayment_rate=parameters["decayment_rate"],
                      episodes=parameters["episodes"],
                      max_steps=parameters["max_steps"],
                      batch_size=parameters["batch_size"])

    dqn_learner.train()

    rewards.append(dqn_learner.rewards_greedy)
    steps.append(dqn_learner.steps_greedy)

rewards = np.array(rewards)
steps = np.array(steps)

saver.save_to_pickle("results/dqn_rewards.pck", rewards)
saver.save_to_pickle("results/dqn_steps.pck", steps)
