from matplotlib import rcParams
import matplotlib.pyplot as plt
from frozenlake.utils import Saver
import numpy as np
import matplotlib.image as mpimg

saver = Saver()

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 6

figure, axes = plt.subplots(nrows=2, ncols=3)

# 0 - ENVS
img = mpimg.imread("figures/envs.PNG")

axes[0, 0].imshow(img)
axes[0, 0].set_ylabel("(a)", labelpad=22)
axes[0, 0].get_xaxis().set_ticks([])
axes[0, 0].get_yaxis().set_ticks([])

axes[0, 0].spines["top"].set_visible(False)
axes[0, 0].spines["right"].set_visible(False)
axes[0, 0].spines["bottom"].set_visible(False)
axes[0, 0].spines["left"].set_visible(False)

# 1 - FROZEN LAKE
medians_q_learning = saver.load_from_pickle("frozenlake/results/q_learning_medians.pickle")
frob_errors_q_learning = saver.load_from_pickle("frozenlake/results/q_learning_frob_errors.pickle")

colors = ['b', 'r', 'g', 'y']
epsilons = sorted([float(epsilon) for epsilon in medians_q_learning.keys()])

medians_lr_learning = saver.load_from_pickle("frozenlake/results/lr_learning_medians.pickle")
frob_errors_lr_learning = saver.load_from_pickle("frozenlake/results/lr_learning_frob_errors.pickle")

for i in range(len(epsilons)):
    label_q = "ϵ=" + str(epsilons[i]) + " Q-learning"
    label_lr = "ϵ=" + str(epsilons[i]) + " LR-learning"
    axes[0, 1].plot(np.arange(0, len(medians_q_learning[str(epsilons[i])]), 5),
                    medians_q_learning[str(epsilons[i])][1::5],
                    c=colors[i],
                    label=label_q,
                    linestyle=(0, (5, 8)),
                    linewidth=0.7)
    axes[0, 1].plot(medians_lr_learning[str(epsilons[i])],
                    c=colors[i],
                    label=label_lr,
                    linewidth=0.7)
axes[0, 1].legend(prop={"size": 4})
axes[0, 1].set_xlim([0, 10000])
axes[0, 1].set_xlabel("Episodes")
axes[0, 1].set_ylabel("(b) Nº of steps")
axes[0, 1].grid(True)

for i in range(len(epsilons)):
    label_q = "ϵ=" + str(epsilons[i]) + " Q-learning"
    label_lr = "ϵ=" + str(epsilons[i]) + " LR-learning"
    axes[0, 2].plot(frob_errors_q_learning[str(epsilons[i])], c=colors[i], label=label_q, linestyle='dashed', linewidth=0.7)
    axes[0, 2].plot(frob_errors_lr_learning[str(epsilons[i])], c=colors[i], label=label_lr, linewidth=0.7)
axes[0, 2].legend(prop={"size": 4})
axes[0, 2].set_xlim([0, 10000])
axes[0, 2].set_xlabel("Episodes")
axes[0, 2].set_ylabel("(c) SFE")
axes[0, 2].grid(True)


# 2 - INVERTED PENDULUM

steps_q_large = saver.load_from_pickle("pendulum/results/exp_1_q_learning_steps.pickle")
final_mean_reward_q_large = saver.load_from_pickle("pendulum/results/exp_1_q_learning_final_reward.pickle")

steps_q_small = saver.load_from_pickle("pendulum/results/exp_2_q_learning_steps.pickle")
final_mean_reward_q_small = saver.load_from_pickle("pendulum/results/exp_2_q_learning_final_reward.pickle")

steps_lr = saver.load_from_pickle("pendulum/results/exp_1_lr_learning_steps.pickle")
final_mean_reward_lr = saver.load_from_pickle("pendulum/results/exp_1_lr_learning_final_reward.pickle")

steps_lr_reg = saver.load_from_pickle("pendulum/results/exp_2_lr_learning_steps.pickle")
final_mean_reward_lr_reg = saver.load_from_pickle("pendulum/results/exp_2_lr_learning_final_reward.pickle")

steps_lr_res = saver.load_from_pickle("pendulum/results/exp_1_lr_res_learning_steps.pickle")
final_mean_reward_lr_res = saver.load_from_pickle("pendulum/results/exp_1_lr_res_learning_final_reward.pickle")

q_large_median_steps = np.median(steps_q_large, axis=0) + 1
q_small_median_steps = np.median(steps_q_small, axis=0) + 1
lr_median_steps = np.median(steps_lr, axis=0) + 1
lr_reg_median_steps = np.median(steps_lr_reg, axis=0) + 1
lr_res_median_steps = np.median(steps_lr_res, axis=0) + 1

q_large_mean_final_rewards = np.median(final_mean_reward_q_large)
q_small_mean_final_rewards = np.median(final_mean_reward_q_small)
lr_mean_final_rewards = np.median(final_mean_reward_lr)
lr_reg_mean_final_rewards = np.median(final_mean_reward_lr_reg)
lr_res_mean_final_rewards = np.median(final_mean_reward_lr_res)

size = len(q_large_median_steps)*100

legend = ["Q-learning - 86,961 params.",
          "Q-learning - 10,605 params.",
          "LR(rank 3) - 6,486 params.",
          "LR reg. - 10,810 params.",
          "LR res. (rank 10) - 5,900 params"]

colors = ["b", "r", "g", "k", "y"]

steps = [q_large_median_steps,
		 q_small_median_steps,
		 lr_median_steps,
		 lr_reg_median_steps,
		 lr_res_median_steps]

for i in range(len(colors)):
    axes[1, 0].plot(np.arange(0, size, 100), steps[i], c=colors[i], linewidth=0.7)
axes[1, 0].axhline(y=100, c="y", linewidth=1)
axes[1, 0].set_xlim(0, 25000)
axes[1, 0].legend(legend, prop={"size": 4})
axes[1, 0].set_xlabel("Episodes")
axes[1, 0].set_ylabel("(d) Median nº of steps")
axes[1, 0].grid(True)

final_rewards = [q_large_mean_final_rewards,
				 q_small_mean_final_rewards,
				 lr_mean_final_rewards,
           		 lr_reg_mean_final_rewards,
           		 lr_res_mean_final_rewards]

axes[1, 1].grid(True, axis='y')
axes[1, 1].bar(x=np.arange(len(final_rewards)), height=np.abs(final_rewards), alpha=.6, color=colors)
cs = {legend[i]: colors[i] for i in range(len(colors))}
labels = list(cs.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=cs[label], alpha=.6) for label in labels]
axes[1, 1].legend(handles, labels, prop={"size": 4})
axes[1, 1].set_ylabel("(e) Cost (negative reward)")


# 3 - ACROBOT
rewards_dqn_light = saver.load_from_pickle("acrobot/results/rewards_1_layer_2000_light.pck")
rewards_dqn_large = saver.load_from_pickle("acrobot/results/rewards_1_layer_2000_large.pck")
rewards_lr = saver.load_from_pickle("acrobot/results/rewards_k_2.pck")
rewards_lr_norm = saver.load_from_pickle("acrobot/results/rewards_k_2_norm.pck")

median_rewards_dqn_light = np.median(rewards_dqn_light, axis=0)
median_rewards_dqn_large = np.median(rewards_dqn_large, axis=0)
median_reward_lr = np.median(rewards_lr, axis=0)
median_reward_lr_norm = np.median(rewards_lr_norm, axis=0)

axes[1, 2].plot(np.arange(0, 5000, 10), median_rewards_dqn_light, 'b', linewidth=0.7)
axes[1, 2].plot(np.arange(0, 5000, 10), median_rewards_dqn_large, 'g', linewidth=0.7)
axes[1, 2].plot(np.arange(0, 5000, 10), median_reward_lr, 'r', linewidth=0.7)
axes[1, 2].plot(np.arange(0, 5000, 10), median_reward_lr_norm, 'y', linewidth=0.7)
axes[1, 2].legend(["DQN mini-batch S=1 - 20,003 params.",
                   "DQN mini-batch S=12 - 20,003 params.",
                   "LR - 18,078 params.",
                   "LR norm. - 18,078 params."], prop={"size": 4})
axes[1, 2].grid(True)
axes[1, 2].set_xlim(0, 5000)
axes[1, 2].set_ylabel("(f) Cumulative reward")
axes[1, 2].set_xlabel("Episodes")

plt.show()
