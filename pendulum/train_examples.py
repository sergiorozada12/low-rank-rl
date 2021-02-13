from utils import Mapper, QLearning, LowRankLearning, Saver, PendulumEnv

mapping = Mapper()
env = PendulumEnv()
saver = Saver()

step = .1
decimal = 1
episodes = 30000
max_steps = 100
alpha_q = .1
alpha_lr = .005
gamma = .9
epsilon = .2
k = 5
lambda_l = .1
lambda_r = .1

state_map, state_reverse_map = mapping.get_state_map(step, decimal)
action_map, action_reverse_map = mapping.get_action_map(step, decimal)

n_states = len(state_map)
n_actions = len(action_map)

q_learner = QLearning(env=env,
                      state_map=state_map,
                      action_map=action_map,
                      state_reverse_map=state_reverse_map,
                      action_reverse_map=action_reverse_map,
                      n_states=n_states,
                      n_actions=n_actions,
                      decimal_state=decimal,
                      decimal_action=decimal,
                      step_state=step,
                      step_action=step,
                      episodes=episodes,
                      max_steps=max_steps,
                      epsilon=epsilon,
                      alpha=alpha_q,
                      gamma=gamma)

lr_learner = LowRankLearning(env=env,
                             state_map=state_map,
                             action_map=action_map,
                             state_reverse_map=state_reverse_map,
                             action_reverse_map=action_reverse_map,
                             n_states=n_states,
                             n_actions=n_actions,
                             decimal_state=decimal,
                             decimal_action=decimal,
                             step_state=step,
                             step_action=step,
                             episodes=episodes,
                             max_steps=max_steps,
                             epsilon=epsilon,
                             alpha=alpha_lr,
                             gamma=gamma,
                             k=k,
                             lambda_l=lambda_l,
                             lambda_r=lambda_r)

q_learner.train()
lr_learner.train()

saver.save_to_pickle("results/q_learner_example.pickle", q_learner)
saver.save_to_pickle("results/low_rank_learner_example.pickle", lr_learner)
