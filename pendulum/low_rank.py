import json
import numpy as np
from utils import Mapper, LowRankLearning, Saver, PendulumEnv

parameters_file = "experiments/exp_2_lr_learning.json"
with open(parameters_file) as j:
    parameters = json.loads(j.read())

mapping = Mapper()
env = PendulumEnv()
saver = Saver()

state_map, state_reverse_map = mapping.get_state_map(parameters["step_state"], parameters["decimal_state"])
action_map, action_reverse_map = mapping.get_action_map(parameters["step_action"], parameters["decimal_action"])

n_states = len(state_map)
n_actions = len(action_map)

steps = []
rewards = []
final_mean_reward = []

for i in range(parameters["n_simulations"]):
    lr_learner = LowRankLearning(env=env,
                                 state_map=state_map,
                                 action_map=action_map,
                                 state_reverse_map=state_reverse_map,
                                 action_reverse_map=action_reverse_map,
                                 n_states=n_states,
                                 n_actions=n_actions,
                                 decimal_state=parameters["decimal_state"],
                                 decimal_action=parameters["decimal_action"],
                                 step_state=parameters["step_state"],
                                 step_action=parameters["step_action"],
                                 episodes=parameters["episodes"],
                                 max_steps=parameters["max_steps"],
                                 epsilon=parameters["epsilon"],
                                 alpha=parameters["alpha"],
                                 gamma=parameters["gamma"],
                                 k=parameters["k"],
                                 lambda_l=parameters["lambda_l"],
                                 lambda_r=parameters["lambda_r"])

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

saver.save_to_pickle("results/exp_2_lr_learning_steps.pickle", steps)
saver.save_to_pickle("results/exp_2_lr_learning_rewards.pickle", rewards)
saver.save_to_pickle("results/exp_2_lr_learning_final_reward.pickle", final_mean_reward)
