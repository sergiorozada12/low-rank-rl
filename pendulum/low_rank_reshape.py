import json
import numpy as np
from utils import MapperReshape, LowRankReshape, Saver, PendulumEnv

parameters_file = "experiments/exp_1_lr_res_learning.json"
with open(parameters_file) as j:
    parameters = json.loads(j.read())

mapping = MapperReshape()
env = PendulumEnv()
saver = Saver()

steps = []
rewards = []
final_mean_reward = []

for i in range(parameters["n_simulations"]):
    lr_learner = LowRankReshape(env=env,
                                k=parameters["k"],
                                mapper=mapping,
                                episodes=parameters["episodes"],
                                max_steps=parameters["max_steps"],
                                epsilon=parameters["epsilon"],
                                alpha=parameters["alpha"],
                                gamma=parameters["gamma"])

    lr_learner.train()

    rs = []
    ss = []

    for j in range(parameters["n_greedy_episodes"]):
        r, s = lr_learner.run_greedy(parameters["n_greedy_steps"])
        rs.append(r)
        ss.append(s)
    r = np.mean(rs)
    s = np.mean(ss)

    steps.append(lr_learner.greedy_steps)
    rewards.append(lr_learner.greedy_r)
    final_mean_reward.append(r)

saver.save_to_pickle("results/exp_1_lr_res_learning_steps.pickle", steps)
saver.save_to_pickle("results/exp_1_lr_res_learning_rewards.pickle", rewards)
saver.save_to_pickle("results/exp_1_lr_res_learning_final_reward.pickle", final_mean_reward)
