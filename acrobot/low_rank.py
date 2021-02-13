import json
import numpy as np
from utils import Mapper, LowRankTD, Saver

parameters_file = "experiments/exp_lr_learning.json"
with open(parameters_file) as j:
    parameters = json.loads(j.read())

mapper = Mapper()
env = gym.make('Acrobot-v1')
env._max_episode_steps = np.inf
saver = Saver()

rewards = []
steps = []

for _ in range(parameters["n_simulations"]):

    low_rank_learner = LowRankTD(env=env,
                                 k=parameters["k"],
                                 mapper=mapper,
                                 episodes=parameters["episodes"],  # 50000,
                                 max_steps=parameters["max_steps"],
                                 epsilon=parameters["epsilon"],
                                 decay=parameters["decay"],
                                 alpha=parameters["alpha"],
                                 gamma=parameters["gamma"])

    low_rank_learner.train()

    rewards.append(low_rank_learner.rewards_greedy)
    steps.append(low_rank_learner.steps_greedy)

rewards = np.array(rewards)
steps = np.array(steps)

saver.save_to_pickle("results/lr_rewards.pck", rewards)
saver.save_to_pickle("results/lr_steps.pck", steps)
