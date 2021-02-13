from matplotlib import rcParams
import matplotlib.pyplot as plt
from utils import Saver
import numpy as np

saver = Saver()

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 16

rewards_dqn_light = saver.load_from_pickle("results/rewards_1_layer_2000_light.pck")
steps_dqn_light = saver.load_from_pickle("results/steps_1_layer_2000_light.pck")

rewards_dqn_large = saver.load_from_pickle("results/rewards_1_layer_2000_large.pck")
steps_dqn_large = saver.load_from_pickle("results/steps_1_layer_2000_large.pck")

rewards_lr = saver.load_from_pickle("results/rewards_k_2.pck")
steps_lr = saver.load_from_pickle("results/steps_k_2.pck")

rewards_lr_norm = saver.load_from_pickle("results/rewards_k_2_norm.pck")
steps_lr_norm = saver.load_from_pickle("results/steps_k_2_norm.pck")

median_rewards_dqn_light = np.median(rewards_dqn_light, axis=0)
median_steps_dqn_light = np.median(steps_dqn_light, axis=0)

median_rewards_dqn_large = np.median(rewards_dqn_large, axis=0)
median_steps_dqn_large = np.median(steps_dqn_large, axis=0)

median_reward_lr = np.median(rewards_lr, axis=0)
median_steps_lr = np.median(steps_lr, axis=0)

median_reward_lr_norm = np.median(rewards_lr_norm, axis=0)
median_steps_lr_norm = np.median(steps_lr_norm, axis=0)

plt.figure(figsize=[6, 4])
plt.plot(np.arange(0, 5000, 10), median_rewards_dqn_light, 'b')
plt.plot(np.arange(0, 5000, 10), median_rewards_dqn_large, 'g')
plt.plot(np.arange(0, 5000, 10), median_reward_lr, 'r')
plt.plot(np.arange(0, 5000, 10), median_reward_lr_norm, 'y')
plt.legend(["DQN mini-batch S=1 - 20,003 params.",
            "DQN mini-batch S=12 - 20,003 params.",
            "LR - 18,078 params.",
            "LR norm. - 18,078 params."], prop={"size": 12})
plt.grid()
plt.xlim(0, 5000)
plt.ylabel("(f) Cumulative reward", size=16)
plt.xlabel("Episodes", size=16)
plt.show()

plt.figure(figsize=[6, 4])
plt.plot(np.arange(0, 5000, 10), median_steps_dqn_light, 'b')
plt.plot(np.arange(0, 5000, 10), median_steps_dqn_large, 'g')
plt.plot(np.arange(0, 5000, 10), median_steps_lr, 'r')
plt.plot(np.arange(0, 5000, 10), median_steps_lr_norm, 'y')
plt.legend(["DQN mini-batch S=1 - 20,003 params.",
            "DQN mini-batch S=12 - 20,003 params.",
            "LR - 18,078 params.",
            "LR norm. - 18,078 params."], prop={"size": 12})
plt.grid()
plt.xlim(0, 5000)
plt.ylabel("Nº of steps", size=16)
plt.xlabel("Episodes", size=16)
plt.show()
