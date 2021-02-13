from utils import QLearning, Saver, get_env

env = get_env()
saver = Saver()

q_learner = QLearning(env=env,
                      episodes=10000,
                      max_steps=100,
                      epsilon=.4,
                      alpha=.9,
                      gamma=.9)

q_learner.train()
saver.save_to_pickle("results/Q_optimal.pickle", q_learner.Q)