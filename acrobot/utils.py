import numpy as np
import matplotlib.pyplot as plt
import PIL
import itertools
import pickle
import gym
from gym import spaces
from gym.utils import seeding


class LowRankTD:
    def __init__(self,
                 env,
                 mapper,
                 k,
                 episodes=100000,
                 max_steps=1000,
                 epsilon=.9,
                 decay=.99999,
                 alpha=.9,
                 gamma=.9):

        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param mapper: Class
            Mapper class that helps implementing the env.
        :param k: int
            Dimension of the latent space.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param decay: float
            Decayment rate of epsilon.
        :param alpha: float
            Learning rate.
        :param gamma: float
            Discount factor.
        """

        self.env = env
        self.mapper = mapper
        self.k = k
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.decay = decay
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [0.0, 1.0, 2.0]

        self.L = np.random.rand(self.mapper.n_rows, k) / 10
        self.R = np.random.rand(k, self.mapper.n_cols) / 10
        self.Q_hat = self.L @ self.R

        self.rewards = []
        self.steps = []
        self.rewards_greedy = []
        self.steps_greedy = []
        self.l_norms = []
        self.r_norms = []

    def get_max_action(self, state):
        """
        :param state: np.array
            Environment state.
        :return int
            Action to take.
        """

        qs = []
        for a in self.actions:
            idx = self.mapper.get_matrix_index(state, a)
            qs.append(self.Q_hat[idx])
        return np.argmax(qs)

    def choose_action(self, state):
        """
        :param state: np.array
            Environment state.
        :return int
            Action to take.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        return self.get_max_action(state)

    def train(self):

        for episode in range(self.episodes):

            s = self.env.reset()
            cumm_reward = 0

            for step in range(self.max_steps):

                a = self.choose_action(s)

                s_prime, r, done, _ = self.env.step(a)

                theta_0 = np.arctan(s_prime[1] / s_prime[0])
                theta_1 = np.arctan(s_prime[3] / s_prime[2])
                r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10
                if done:
                    r += 100

                cumm_reward += r

                mat_idx = mapper.get_matrix_index(s, a)
                mat_idx_prime = mapper.get_matrix_index(s_prime, self.get_max_action(s_prime))

                q_h = self.Q_hat[mat_idx]
                q_b = r + self.gamma * self.Q_hat[mat_idx_prime]

                self.L[mat_idx[0], :] += self.alpha * (q_b - q_h) * self.R[:, mat_idx[
                    1]]/np.linalg.norm(self.R[:, mat_idx[1]])
                self.R[:, mat_idx[1]] += self.alpha * (q_b - q_h) * self.L[mat_idx[0],
                                                                    :]/np.linalg.norm(self.L[mat_idx[0], :])

                s = s_prime

                self.Q_hat[mat_idx[0], :] = self.L[mat_idx[0], :] @ self.R
                self.Q_hat[:, mat_idx[1]] = self.L @ self.R[:, mat_idx[1]]

                self.epsilon *= self.decay

                if done:
                    break

            self.rewards.append(cumm_reward)
            self.steps.append(step)
            self.l_norms.append(np.linalg.norm(self.L, 'fro'))
            self.r_norms.append(np.linalg.norm(self.R, 'fro'))

            if (episode % 10) == 0:
                r, s = self.run_greedy(self.max_steps)
                self.rewards_greedy.append(r)
                self.steps_greedy.append(s)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset()
        cum_r = 0

        for step in range(n_steps):

            a = self.get_max_action(s)
            s_prime, r, done, info = self.env.step(a)

            theta_0 = np.arctan(s_prime[1] / s_prime[0])
            theta_1 = np.arctan(s_prime[3] / s_prime[2])

            r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10
            if done:
                r += 100

            cum_r += r

            if done:
                break

            s = s_prime

        return cum_r, step

    def test(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset()

        for _ in range(n_steps):

            a = self.get_max_action(s)
            s_prime, r, done, info = self.env.step(a)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))

            if done:
                break

            s = s_prime

        self.env.close()


class Buffer:
    def __init__(self, size):
        """
        :param size: int
            Size of the buffer.
        """

        self.buffer = []
        self.limit = size

    def sample_buffer(self, n):
        """
        :param n: int
            Number of samples to take.
        :return: array
            Array of samples of experiences.
        """

        indices = np.random.randint(0, len(self.buffer), size=n)
        return np.array([self.buffer[index] for index in indices])

    def store_buffer(self, item):
        """
        :param item: tuple
            Tuple of state-action-reward-state.
        """

        if len(self.buffer) > self.limit:
            self.buffer.pop(0)

        self.buffer.append(item)


class Dqn:
    def __init__(self,
                 env,
                 model,
                 buffer,
                 episodes=5000,
                 max_steps=1000,
                 gamma=0.99,
                 epsilon=1.0,
                 decayment_rate=0.99999,
                 batch_size=12):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param model: Keras.models
            NN Function approximator.
        :param buffer: Buffer
            Buffer to store experiences.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps per episode.
        :param gamma: float
            Discount factor.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param decayment_rate: float
            Decayment rate of epsilon.
        :param batch_size: int
            Size of the training batch.
        """

        self.model = model
        self.buffer = buffer
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decayment_rate = decayment_rate
        self.episodes = episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.rewards = []
        self.steps = []
        self.rewards_greedy = []
        self.steps_greedy = []

    def choose_action(self, st):
        """
        :param state: np.array
            Environment state.
        :return int
            Action to take.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(st)[0])

    def train(self):

        for episode in range(self.episodes):

            cum_r = 0
            s = self.env.reset()
            s = np.reshape(s, [1, self.n_states])

            for step in range(self.max_steps):

                a = self.choose_action(s)
                s_prime, r, done, info = self.env.step(a)
                theta_0 = np.arctan(s_prime[1] / s_prime[0])
                theta_1 = np.arctan(s_prime[3] / s_prime[2])
                r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10

                if done:
                    r += 100

                cum_r += r

                s_prime = np.reshape(s_prime, [1, self.n_states])
                self.buffer.store_buffer([s[0], s_prime[0], a, r])

                if episode > 1:
                    snap = self.buffer.sample_buffer(self.batch_size)
                    st, nst, at, rt = np.vstack(snap[:, 0]), np.vstack(snap[:, 1]), snap[:, 2], snap[:, 3]
                    target_action = rt + self.gamma * np.amax(self.model.predict(nst), axis=1)
                    target = self.model.predict(st)
                    for i in range(self.batch_size):
                        target[i, at[i]] = target_action[i]
                    self.model.fit(st, target, verbose=False)

                s = s_prime

                self.epsilon *= self.decayment_rate

                if done:
                    break

            self.rewards.append(cum_r)
            self.steps.append(step)

            if (episode % 10) == 0:
                greedy_r, greedy_steps = self.run_greedy(self.model, self.max_steps)
                self.rewards_greedy.append(greedy_r)
                self.steps_greedy.append(greedy_steps)

    def run_greedy(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset()
        s = np.reshape(s, [1, self.n_states])
        cum_r = 0

        for i in range(n_steps):
            a = np.argmax(self.model.predict(s)[0])
            s_prime, r, done, info = self.env.step(a)

            theta_0 = np.arctan(s_prime[1] / s_prime[0])
            theta_1 = np.arctan(s_prime[3] / s_prime[2])
            r = -(-1 - np.cos(theta_0) - np.cos(theta_1 + theta_0)) ** 2 / 10

            if done:
                r += 100

            s_prime = np.reshape(s_prime, [1, self.n_states])

            cum_r += r
            s = s_prime

            if done:
                break

        return cum_r, i

    def test(self, n_steps):
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        """

        state = self.env.reset()
        state = np.reshape(state, [1, self.n_states])

        for _ in range(n_steps):
            action = np.argmax(self.model.predict(state)[0])
            new_state, reward, done, info = self.env.step(action)
            PIL.Image.fromarray(self.env.render(mode='rgb_array')).resize((320, 420))
            state = new_state
            state = np.reshape(state, [1, self.n_states])

            if done:
                break

        self.env.close()


class Mapper:
    def __init__(self):

        cos_theta_1 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        sin_theta_1 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        cos_theta_2 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        sin_theta_2 = np.around(np.arange(-1.0, 1.0 + 0.2, 0.2), 1) + 0.
        theta_dot_1 = np.around(np.arange(-14.0, 14.0 + 2.0, 2.0), 1) + 0.
        theta_dot_2 = np.around(np.arange(-30.0, 30.0 + 2.0, 2.0), 1) + 0.
        push = np.around(np.arange(0.0, 2.0 + 1.0, 1.0), 1) + 0.

        grid = [cos_theta_1, sin_theta_1, cos_theta_2, sin_theta_2, theta_dot_1, theta_dot_2, push]

        self.forward_map = [combination for combination in itertools.product(*grid)]
        random.shuffle(self.forward_map)

        self.n_rows = int(np.ceil(np.sqrt(len(self.forward_map))))
        self.n_cols = self.n_rows - 1
        matrix_idx = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                matrix_idx.append((i, j))

        self.reverse_map = dict(zip(self.forward_map, matrix_idx))

    def get_matrix_index(self, state, action):
        """
        :param state: np.array
            Array containing the state
        :param action: int
            Action
        :return: tuple
            Index of the matrix
        """

        cos_th_1 = -1.0 + 0.2 * (state[0] + 1.0) // 0.2
        sin_th_1 = -1.0 + 0.2 * (state[1] + 1.0) // 0.2
        cos_th_2 = -1.0 + 0.2 * (state[2] + 1.0) // 0.2
        sin_th_2 = -1.0 + 0.2 * (state[3] + 1.0) // 0.2
        th_dot_1 = -14 + 2 * ((state[4] + 14) // 2)
        th_dot_2 = -30 + 2 * ((state[5] + 30) // 2)

        key = (cos_th_1, sin_th_1, cos_th_2, sin_th_2, th_dot_1, th_dot_2, action)
        return self.reverse_map[key]


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
