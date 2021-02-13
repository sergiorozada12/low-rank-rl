import numpy as np
import matplotlib.pyplot as plt
import PIL
import itertools
import pickle
import gym
from gym import spaces
from gym.utils import seeding


class QLearning:
    def __init__(self,
                 env,
                 state_map,
                 action_map,
                 state_reverse_map,
                 action_reverse_map,
                 n_states,
                 n_actions,
                 decimal_state,
                 decimal_action,
                 step_state,
                 step_action,
                 episodes=100000,
                 max_steps=1000,
                 epsilon=.2,
                 alpha=.9,
                 gamma=.9):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param state_map: dict
            Dictionary with the index of the state as key and the codified state as value.
        :param action_map: dict
            Dictionary with the index of the action as key and the codified action as value.
        :param state_reverse_map: dict
            Dictionary with codified state as key and the corresponding index as value.
        :param action_reverse_map: dict
            Dictionary with codified action as key and the corresponding index as value.
        :param n_states: int
            Number of states.
        :param n_actions:  int
            Number of actions.
        :param decimal_state: float
            Precision of the state.
        :param decimal_action: float
            Precision of the action
        :param step_state: float
            Step of the state discretization.
        :param step_action: float
            Step of the action discretization.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param alpha: float
            Learning rate.
        :param gamma: float
            Discount factor.
        """

        self.env = env
        self.state_map = state_map
        self.action_map = action_map
        self.state_reverse_map = state_reverse_map
        self.action_reverse_map = action_reverse_map
        self.n_states = n_states
        self.n_actions = n_actions
        self.step_state = step_state
        self.step_action = step_action
        self.decimal_state = decimal_state
        self.decimal_action = decimal_action
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions))

        self.steps = []
        self.cummulative_reward = []
        self.greedy_r = []
        self.greedy_steps = []

    def get_s_idx(self, st):
        """
        :param st: np.array
            State to obtain the row index of the Q-matrix
        :return: int
            Row index of the Q-matrix
        """

        st_ = np.array([np.arctan(st[1] / st[0]), st[2]])
        st_ = [self.step_state * (np.round(s / self.step_state)) for s in st_]
        return self.state_reverse_map[str(np.around(st_, self.decimal_state) + 0.)]

    def get_a_idx(self, at):
        """
        param at: np.array
            Action to obtain the column index of the Q-matrix
        :return: int
            Column index of the Q-matrix
        """

        at_ = [self.step_action * (np.round(a / self.step_action)) for a in at]
        return self.action_reverse_map[str(np.around(at_, self.decimal_action) + 0.)]

    def choose_action(self, st_idx):
        """
        :param st_idx: int
            Row index of the current state.
        :return tuple
            Selected action and column index of the selected action.
        """

        if np.random.rand() < self.epsilon:
            a = self.env.action_space.sample()
            return a, self.get_a_idx(a)
        a_idx = np.argmax(self.Q[st_idx, :])
        return self.action_map[a_idx], a_idx

    def train(self):

        for episode in range(self.episodes):
            s = self.env.reset(upright=True)
            s_idx = self.get_s_idx(s)

            cumm_reward = 0

            for step in range(self.max_steps):
                a, a_idx = self.choose_action(s_idx)

                s_prime, r, _, _ = self.env.step(a)
                theta = np.arctan(s_prime[1] / s_prime[0])
                done = True if ((theta > np.pi / 4) | (theta < -np.pi / 4)) else False
                r = .1 - (np.arccos(s[0]) ** 2 + s[2] ** 2 + a[0] ** 2)
                s_prime_idx = self.get_s_idx(s_prime)
                cumm_reward += r

                target_q = r + self.gamma * np.max(self.Q[s_prime_idx, :])
                error_signal = target_q - self.Q[s_idx, a_idx]
                self.Q[s_idx, a_idx] += self.alpha * error_signal

                s = s_prime
                s_idx = s_prime_idx

                if done:
                    break

            self.cummulative_reward.append(cumm_reward)
            self.steps.append(step)

            if (episode % 100) == 0:
                greedy_r, greedy_steps = self.run_greedy(self.max_steps)
                self.greedy_r.append(greedy_r)
                self.greedy_steps.append(greedy_steps)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset(upright=True)
        cum_r = 0

        for i in range(n_steps):
            s_idx = self.get_s_idx(s)
            a_idx = np.argmax(self.Q[s_idx, :])
            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            theta = np.arctan(s_prime[1] / s_prime[0])
            done = (theta > 1.0) | (theta < -1.0)
            cum_r += .1 - (np.arccos(s[0]) ** 2 + s[2] ** 2 + a[0] ** 2)
            s = s_prime

            if done:
                break

        return cum_r, i

    def test(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the test episode.
        """

        s = self.env.reset(upright=True)

        for i in range(n_steps):
            s_idx = self.get_s_idx(s)
            a_idx = np.argmax(self.Q[s_idx, :])
            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))
            s = s_prime

            theta = np.arctan(s[1] / s[0])
            if (theta > 1.0) | (theta < -1.0):
                break
        self.env.close()


class LowRankLearning:
    def __init__(self,
                 env,
                 state_map,
                 action_map,
                 state_reverse_map,
                 action_reverse_map,
                 n_states,
                 n_actions,
                 decimal_state,
                 decimal_action,
                 step_state,
                 step_action,
                 episodes=100000,
                 max_steps=1000,
                 epsilon=.2,
                 alpha=.005,
                 gamma=.9,
                 k=5,
                 lambda_l=0.0,
                 lambda_r=0.0):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param state_map: dict
            Dictionary with the index of the state as key and the codified state as value.
        :param action_map: dict
            Dictionary with the index of the action as key and the codified action as value.
        :param state_reverse_map: dict
            Dictionary with codified state as key and the corresponding index as value.
        :param action_reverse_map: dict
            Dictionary with codified action as key and the corresponding index as value.
        :param n_states: int
            Number of states.
        :param n_actions:  int
            Number of actions.
        :param decimal_state: float
            Precision of the state.
        :param decimal_action: float
            Precision of the action
        :param step_state: float
            Step of the state discretization.
        :param step_action: float
            Step of the action discretization.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param alpha: float
            Learning rate.
        :param gamma: float
            Discount factor.
        :param k: int
            Dimension of the latent space.
        :param lambda_l: float
            Regularizer of the left Frobenius norm.
        :param lambda_r: float
            Regularizer of the right Frobenius norm.
        """

        self.env = env
        self.state_map = state_map
        self.action_map = action_map
        self.state_reverse_map = state_reverse_map
        self.action_reverse_map = action_reverse_map
        self.n_states = n_states
        self.n_actions = n_actions
        self.step_state = step_state
        self.step_action = step_action
        self.decimal_state = decimal_state
        self.decimal_action = decimal_action
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        self.lambda_l = lambda_l
        self.lambda_r = lambda_r

        self.L = np.random.rand(n_states, k)
        self.R = np.random.rand(k, n_actions)

        self.cummulative_reward = []
        self.steps = []
        self.greedy_r = []
        self.greedy_steps = []

    def get_s_idx(self, st):
        """
        :param st: np.array
            State to obtain the row index of the Q-matrix
        :return: int
            Row index of the Q-matrix
        """

        st_ = np.array([np.arctan(st[1] / st[0]), st[2]])
        st_ = [self.step_state * (np.round(s / self.step_state)) for s in st_]
        return self.state_reverse_map[str(np.around(st_, self.decimal_state) + 0.)]

    def get_a_idx(self, at):
        """
        param at: np.array
            Action to obtain the column index of the Q-matrix
        :return: int
            Column index of the Q-matrix
        """

        at_ = [self.step_action * (np.round(a / self.step_action)) for a in at]
        return self.action_reverse_map[str(np.around(at_, self.decimal_action) + 0.)]

    def choose_action(self, q_current_state):
        """
        :param q_current_state: np.array
            Q-values corresponding to the current state and all possible actions.
        :return tuple
            Selected action and column index of the selected action.
        """

        if np.random.rand() < self.epsilon:
            a = self.env.action_space.sample()
            return a, self.get_a_idx(a)
        a_idx = np.argmax(q_current_state)
        return self.action_map[a_idx], a_idx

    def train(self):

        for episode in range(self.episodes):

            s = self.env.reset(upright=True)
            s_idx = self.get_s_idx(s)
            q_current_state = self.L[s_idx, :] @ self.R

            cumm_reward = 0

            for step in range(self.max_steps):
                a, a_idx = self.choose_action(q_current_state)

                s_prime, r, _, _ = self.env.step(a)
                theta = np.arctan(s_prime[1] / s_prime[0])
                done = True if ((theta > np.pi / 4) | (theta < -np.pi / 4)) else False
                r = .1 - (np.arccos(s[0]) ** 2 + s[2] ** 2 + a[0] ** 2)
                s_prime_idx = self.get_s_idx(s_prime)
                cumm_reward += r

                q_next_state = self.L[s_prime_idx, :] @ self.R
                q_bootstrapped = r + self.gamma * np.max(q_next_state)
                q_hat = q_current_state[a_idx]

                err = (q_bootstrapped - q_hat)

                self.L[s_idx, :] -= self.alpha * (-err * self.R[:, a_idx] + self.lambda_l * self.L[s_idx, :])
                self.R[:, a_idx] -= self.alpha * (-err * self.L[s_idx, :] + self.lambda_r * self.R[:, a_idx])

                s = s_prime
                s_idx = s_prime_idx
                q_current_state = q_next_state

                if done:
                    break

            self.cummulative_reward.append(cumm_reward)
            self.steps.append(step)

            if (episode % 100) == 0:
                greedy_r, greedy_steps = self.run_greedy(self.max_steps)
                self.greedy_r.append(greedy_r)
                self.greedy_steps.append(greedy_steps)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset(upright=True)
        Q_hat = self.L @ self.R
        cum_r = 0

        for i in range(n_steps):
            s_idx = self.get_s_idx(s)
            a_idx = np.argmax(Q_hat[s_idx, :])
            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            theta = np.arctan(s_prime[1] / s_prime[0])
            done = (theta > 1.0) | (theta < -1.0)
            cum_r += .1 - (np.arccos(s[0]) ** 2 + s[2] ** 2 + a[0] ** 2)
            s = s_prime

            if done:
                break

        return cum_r, i

    def test(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the test episode.
        """

        s = self.env.reset(upright=True)
        Q_hat = self.L @ self.R

        for i in range(n_steps):
            s_idx = self.get_s_idx(s)
            a_idx = np.argmax(Q_hat[s_idx, :])
            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))
            s = s_prime

            theta = np.arctan(s[1] / s[0])
            if (theta > 1.0) | (theta < -1.0):
                pass

        self.env.close()


class LinearLearning:
    def __init__(self,
                 env,
                 state_set,
                 action_set,
                 state_map,
                 state_reverse_map,
                 action_map,
                 action_reverse_map,
                 decimal_state,
                 decimal_action,
                 step_action,
                 step_state,
                 episodes=100000,
                 max_steps=100,
                 epsilon=.9,
                 alpha=.9,
                 gamma=.9):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param state_set: list
            Fixed point states to calculate features.
        :param state_map: dict
            Dictionary with the index of the state as key and the codified state as value.
        :param action_map: dict
            Dictionary with the index of the action as key and the codified action as value.
        :param state_reverse_map: dict
            Dictionary with codified state as key and the corresponding index as value.
        :param action_reverse_map: dict
            Dictionary with codified action as key and the corresponding index as value.
        :param decimal_state: float
            Precision of the state.
        :param decimal_action: float
            Precision of the action
        :param step_state: float
            Step of the state discretization.
        :param step_action: float
            Step of the action discretization.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param alpha: float
            Learning rate.
        :param gamma: float
            Discount factor.
        """

        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.state_map = state_map
        self.state_reverse_map = state_reverse_map
        self.action_map = action_map
        self.action_reverse_map = action_reverse_map
        self.state_set = state_set
        self.action_set = action_set
        self.step_action = step_action
        self.step_state = step_state
        self.decimal_action = decimal_action
        self.decimal_state = decimal_state

        self.size = len(self.state_set) * len(self.action_set) + len(self.action_set)
        self.w = np.random.rand(self.size, 1)/10

        self.cumulative_reward = []
        self.steps = []
        self.greedy_r = []
        self.greedy_steps = []

        self.features = []
        self.features_map = {}
        for i, state in enumerate(self.state_map):
            for j, action in enumerate(self.action_map):
                self.features.append(self.get_features(state, j))
                self.features_map[str(i) + str(j)] = i * len(self.action_map) + j
        self.features = np.array(self.features)

        self.n_states = len(self.state_map)
        self.n_actions = len(self.action_map)

    def choose_action(self, a_idx):
        """
        :param st_idx: int
            Row index of the current state.
        :return tuple
            Selected action and index of the selected action.
        """

        if np.random.rand() < self.epsilon:
            a = self.env.action_space.sample()
            return a, self.get_a_idx(a)
        return self.action_map[a_idx], a_idx

    def get_features(self, s, a):
        """
        :param s: np.array
            Current state.
        :param a: int
            Index of the current action
        :return: np.array
            Features of the state-action pair
        """

        position = a * (len(self.state_set) + 1)
        encoded_state = np.zeros(self.size)

        s_ = np.array([1] + list([np.exp(-np.sum((s - mu) ** 2) / 2) for mu in self.state_set])).reshape(1, -1)

        encoded_state[position:position + s_.shape[1]] = s_
        return encoded_state

    def get_s_idx(self, st):
        """
        :param st: np.array
            State to obtain the row index.
        :return: int
            Index of the state.
        """

        st_ = np.array([np.arctan(st[1] / st[0]), st[2]])
        st_ = [self.step_state * (np.round(s / self.step_state)) for s in st_]
        return self.state_reverse_map[str(np.around(st_, self.decimal_state) + 0.)]

    def get_a_idx(self, at):
        """
        param at: np.array
            Action to obtain the index of the action.
        :return: int
            Index of the action.
        """

        at_ = [self.step_action * (np.round(a / self.step_action)) for a in at]
        return self.action_reverse_map[str(np.around(at_, self.decimal_action) + 0.)]

    def train(self):

        for episode in range(self.episodes):
            s = self.env.reset(upright=True)
            s_idx = self.get_s_idx(s)

            idx = s_idx * self.n_actions
            features_set = self.features[idx:idx + self.n_actions, :]
            a_idx = np.argmax((features_set @ self.w).flatten())

            cumm_reward = 0

            for step in range(self.max_steps):
                a, a_idx = self.choose_action(a_idx)
                s_prime, r, done, info = self.env.step(a)
                s_prime_idx = self.get_s_idx(s_prime)
                theta = np.arctan(s_prime[1] / s_prime[0])
                done = True if ((theta > np.pi / 4) | (theta < -np.pi / 4)) else False
                r = .1 - (np.arccos(s[0]) ** 2 + s[2] ** 2 + a[0] ** 2)

                idx = s_prime_idx * self.n_actions
                features_set = self.features[idx:idx + self.n_actions, :]
                a_prime_idx = np.argmax((features_set @ self.w).flatten())

                features = self.features[s_idx * self.n_actions + a_idx, :]
                features_prime = self.features[s_prime_idx * self.n_actions + a_prime_idx, :]

                q_bootstrapped = r + self.gamma * features_prime @ self.w
                q_hat = features @ self.w
                err = (q_bootstrapped - q_hat)[0]

                self.w -= self.alpha * (-err * features.reshape(-1, 1))

                cumm_reward += r

                s = s_prime
                s_idx = s_prime_idx
                a_idx = a_prime_idx

                if done:
                    break

            if (episode % 100) == 0:
                greedy_r, greedy_steps = self.run_greedy(self.max_steps)
                self.greedy_r.append(greedy_r)
                self.greedy_steps.append(greedy_steps)

            self.cumulative_reward.append(cumm_reward)
            self.steps.append(step)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset(upright=True)
        s_idx = self.get_s_idx(s)

        cum_r = 0

        for i in range(n_steps):
            idx = s_idx * self.n_actions
            features_set = self.features[idx:idx + self.n_actions, :]
            a_idx = np.argmax((features_set @ self.w).flatten())

            a = self.action_map[a_idx]
            s_prime, r, done, info = self.env.step(a)
            s_prime_idx = self.get_s_idx(s_prime)

            theta = np.arctan(s_prime[1] / s_prime[0])
            done = True if ((theta > np.pi / 4) | (theta < -np.pi / 4)) else False
            cum_r += .1 - (np.arccos(s[0]) ** 2 + s[2] ** 2 + a[0] ** 2)

            s = s_prime
            s_idx = s_prime_idx

            if done:
                break

        return cum_r, i

    def test(self, steps):
        """
        :param n_steps: int
            Number of steps of the test episode.
        """

        s = self.env.reset(upright=True)
        s_idx = self.get_s_idx(s)
        plt.imshow(self.env.render(mode='rgb_array'))

        for i in range(steps):
            idx = s_idx * self.n_actions
            features_set = self.features[idx:idx + self.n_actions, :]
            a_idx = np.argmax((features_set @ self.w).flatten())
            a = self.action_map[a_idx]

            s_prime, r, done, info = self.env.step(a)
            s_prime_idx = self.get_s_idx(s_prime)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))

            theta = np.arctan(s_prime[1] / s_prime[0])
            done = True if ((theta > np.pi / 4) | (theta < -np.pi / 4)) else False

            if done:
                break

            s_idx = s_prime_idx
        self.env.close()


class Mapper:
    def __init__(self):

        self.min_theta = -1.0 # 50 grados
        self.max_theta = 1.0 # 50 grados
        self.min_theta_dot = -5.0
        self.max_theta_dot = 5.0

        self.min_joint_effort = -2.0
        self.max_joint_effort = 2.0

    def get_map(self, iterable):
        """
        :param iterable: list
            Elements of the state space.
        :return: dict
            Cartesian product of the state space.
        """

        mapping = [np.array(combination) for combination in itertools.product(*iterable)]
        reverse_mapping = {str(mapping[i]):i for i in range(len(mapping))}

        return mapping, reverse_mapping

    def get_state_map(self, step, decimal):
        """
        :param step: float
            Discretization step.
        :param decimal: int
            Precision.
        :return: dict
            Map of states and indices.
        """

        theta = np.around(np.arange(self.min_theta, self.max_theta + step, step), decimal) + 0.
        theta_dot = np.around(np.arange(self.min_theta_dot, self.max_theta_dot + step, step), decimal) + 0.

        return self.get_map([theta, theta_dot])

    def get_action_map(self, step, decimal):
        """
        :param step: float
            Discretization step.
        :param decimal: int
            Precision.
        :return: dict
            Map of actions and indices.
        """

        joint_effort = np.around(np.arange(self.min_joint_effort, self.max_joint_effort + step, step), decimal) + 0.

        return self.get_map([joint_effort])


class Saver:
    @staticmethod
    def save_to_pickle(path, obj):
        """

        :param path: str
            Path to store the object.
        :param obj
            Object to store.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_from_pickle(path):
        """
        :param path: str
            Path of the object to load.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


class TestUtils:

    @staticmethod
    def smooth_signal(signal, w):
        """
        :param signal: list
            Signal to smooth.
        :param w: int
            Window of the smoothing operation.
        :return: tuple
            Mean and standard deviation of the smoothed signal.
        """

        avg = np.array([np.mean(signal[i:i + w]) for i in range(len(signal) - w)])
        std = np.array([np.std(signal[i:i + w]) for i in range(len(signal) - w)])

        return avg, std

    @staticmethod
    def plot_steps(steps, legend, colors, size):
        """
        :param steps: np.array
            Steps to plot.
        :param legend: list
            List of labels to plot.
        :param colors: list
            List of colors to plot.
        :param size: int
            Size of the x-axis to scale it properly.
        """

        plt.figure(figsize=[6, 4])
        plt.grid()
        for i in range(len(colors)):
            plt.plot(np.arange(0, size, 100), steps[i], c=colors[i])
        plt.xlim(0, 25000)
        plt.legend(legend, prop={"size": 12})
        plt.xlabel("Episodes")
        plt.ylabel("(d) Median nº of steps")
        plt.show()

    @staticmethod
    def plot_smoothed_rewards(medians, stds, legend, colors, size):
        """
        :param medians: list
            Array with the medians of the cum. rewards.
        :param stds: list
            Array with the standard devs. of the cum. rewards.
        :param legend: list
            List of labels to plot.
        :param colors: list
            List of colors to plot.
        :param size: int
            Size of the x-axis to scale it properly.
        :return:
        """

        plt.figure(figsize=[6, 4])
        plt.grid()
        for i in range(len(colors)):
            plt.plot(np.arange(0, size * 100, 100), medians[i], c=colors[i])
            plt.fill_between(np.arange(0, size * 100, 100),
                             medians[i] + stds[i],
                             medians[i] - stds[i],
                             color=colors[i],
                             alpha=.1)
        plt.ylim(-45, -5)
        plt.xlim(0, 25000)
        plt.xlabel("Episodes")
        plt.ylabel("Smoothed cumm. reward")
        plt.legend(legend, prop={"size": 12})
        plt.show()

    @staticmethod
    def plot_final_rewards(rewards, legend, colors):
        """
        :param rewards: list
            List with the mean rewards of the trained agents in several greedy episodes
        :param legend: list
            List of labels to plot.
        :param colors: list
            List of colors to plot.
        """

        plt.figure(figsize=[7, 5])
        plt.grid()
        plt.bar(x=np.arange(len(rewards)), height=np.abs(rewards), alpha=.6, color=colors)
        cs = {legend[i]: colors[i] for i in range(len(colors))}
        labels = list(cs.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=cs[label], alpha=.6) for label in labels]
        plt.legend(handles, labels, prop={"size": 12})
        plt.ylabel("(e) Cost (negative reward)")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.show()

    @staticmethod
    def plot_singular_values(s_q, s_lr, legend):
        plt.figure(figsize=[6, 4])
        plt.grid()
        plt.bar(x=np.arange(len(s_q)), height=s_q, alpha=.6, width=.4, color='b')
        plt.bar(x=np.arange(len(s_lr)) + .4, height=s_lr, alpha=.6, width=.4, color='orange')
        plt.legend(legend, prop={"size": 12})
        plt.ylabel("Normalized singular value")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.show()


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self, upright=False):
        high = np.array([np.pi, 1])
        if upright:
            self.state = [np.random.rand()/100, np.random.rand()/100]
        else:
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = "resources/clockwise.png"
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def angle_normalize(self, x):
        return ((x+np.pi) % (2*np.pi)) - np.pi

