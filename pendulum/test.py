import numpy as np
from utils import Saver, TestUtils

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 16


saver = Saver()
test_utils = TestUtils()

q_learner = saver.load_from_pickle("results/q_learner_example.pickle")
lr_learner = saver.load_from_pickle("results/low_rank_learner_example.pickle")

steps_q_large = saver.load_from_pickle("results/exp_1_q_learning_steps.pickle")
rewards_q_large = saver.load_from_pickle("results/exp_1_q_learning_rewards.pickle")
final_mean_reward_q_large = saver.load_from_pickle("results/exp_1_q_learning_final_reward.pickle")

steps_q_small = saver.load_from_pickle("results/exp_2_q_learning_steps.pickle")
rewards_q_small = saver.load_from_pickle("results/exp_2_q_learning_rewards.pickle")
final_mean_reward_q_small = saver.load_from_pickle("results/exp_2_q_learning_final_reward.pickle")

steps_lr = saver.load_from_pickle("results/exp_1_lr_learning_steps.pickle")
rewards_lr = saver.load_from_pickle("results/exp_1_lr_learning_rewards.pickle")
final_mean_reward_lr = saver.load_from_pickle("results/exp_1_lr_learning_final_reward.pickle")

steps_lr_reg = saver.load_from_pickle("results/exp_2_lr_learning_steps.pickle")
rewards_lr_reg = saver.load_from_pickle("results/exp_2_lr_learning_rewards.pickle")
final_mean_reward_lr_reg = saver.load_from_pickle("results/exp_2_lr_learning_final_reward.pickle")

q_large_median_steps = np.median(steps_q_large, axis=0)
q_small_median_steps = np.median(steps_q_small, axis=0)
lr_median_steps = np.median(steps_lr, axis=0)
lr_reg_median_steps = np.median(steps_lr_reg, axis=0)

q_large_mean_rewards = np.mean(rewards_q_large, axis=0)
q_small_mean_rewards = np.mean(rewards_q_small, axis=0)
lr_mean_rewards = np.mean(rewards_lr, axis=0)
lr_reg_mean_rewards = np.mean(rewards_lr_reg, axis=0)

q_large_mean_final_rewards = np.median(final_mean_reward_q_large)
q_small_mean_final_rewards = np.median(final_mean_reward_q_small)
lr_mean_final_rewards = np.median(final_mean_reward_lr)
lr_reg_mean_final_rewards = np.median(final_mean_reward_lr_reg)

q_large_std_rewards = np.std(rewards_q_large, axis=0)
q_small_std_rewards = np.std(rewards_q_small, axis=0)
lr_std_rewards = np.std(rewards_lr, axis=0)
lr_reg_std_rewards = np.std(rewards_lr_reg, axis=0)

# PLOT 1
size = len(q_large_median_steps)*100

legend = ["Q-learning - 86,961 params.",
          "Q-learning - 10,605 params.",
          "LR(rank 3) - 6,486 params.",
          "LR reg. - 10,810 params."]

colors = ["b", "r", "g", "k"]

steps = [q_large_median_steps,
		 q_small_median_steps,
		 lr_median_steps,
		 lr_reg_median_steps]

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 16

test_utils.plot_steps(steps, legend, colors, size)

#PLOT 2
qs_mu = []
ql_mu = []
lr_mu = []
lrr_mu = []
qs_std = []
ql_std = []
lr_std = []
lrr_std = []

for i in range(len(rewards_q_large)):
	ql_mu_, qs_std_ = test_utils.smooth_signal(rewards_q_large[i], w=50)
	qs_mu_, ql_std_ = test_utils.smooth_signal(rewards_q_small[i], w=50)
	lr_mu_, lr_std_ = test_utils.smooth_signal(rewards_lr[i], w=50)
	lrr_mu_, lrr_std_ = test_utils.smooth_signal(rewards_lr_reg[i], w=50)

	qs_mu.append(qs_mu_)
	ql_mu.append(ql_mu_)
	lr_mu.append(lr_mu_)
	lrr_mu.append(lrr_mu_)

	qs_std.append(qs_std_)
	ql_std.append(ql_std_)
	lr_std.append(lr_std_)
	lrr_std.append(lrr_std_)

median_rewards = [np.median(ql_mu, axis=0),
				  np.median(qs_mu, axis=0),
				  np.median(lr_mu, axis=0),
				  np.median(lrr_mu, axis=0)]

stds_rewards = [np.std(ql_std, axis=0),
				np.std(qs_std, axis=0),
				np.std(lr_std, axis=0),
				np.std(lrr_std, axis=0)]

size = ql_mu[1].shape[0]

test_utils.plot_smoothed_rewards(median_rewards, stds_rewards, legend, colors, size)

# PLOT 3
final_rewards = [q_large_mean_final_rewards,
				 q_small_mean_final_rewards,
				 lr_mean_final_rewards,
           		 lr_reg_mean_final_rewards]

test_utils.plot_final_rewards(final_rewards, legend, colors)

# PLOT 4
u, s_q, vt = np.linalg.svd(q_learner.Q)
u, s_lr, vt = np.linalg.svd(lr_learner.L @ lr_learner.R)

s_q_norm = s_q/np.sum(s_q)
s_lr_norm = s_lr/np.sum(s_lr)

s_q_norm = s_q_norm[:10]
s_lr_norm = s_lr_norm[:10]

legend = ["Q-learning - 86,961 params.", "LR reg. - 10,810 params."]

test_utils.plot_singular_values(s_q_norm, s_lr_norm, legend)
