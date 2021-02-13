import numpy as np
import matplotlib.pyplot as plt
import pickle
import gym
from gym.envs.registration import register

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['font.size'] = 16


class QLearning:

    def __init__(self,
                 env,
                 episodes=1000,
                 max_steps=100,
                 epsilon=.9,
                 alpha=.9,
                 gamma=.9):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps por episode.
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

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.Q_optimal = None

        self.steps = []
        self.frobenius_error = []

    def choose_action(self, s):
        """
        :param s: np.array
            Current state.
        :return int
            Action index.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[s, :])

    def frobenius_norm(self, order):
        """
        :param order: int
            Exponent of the norm.
        :return float
            Squared Frobenius Error (SFE).
        """

        return np.sum((self.Q_optimal.flatten() - self.Q.flatten()) ** order)

    def train(self, check_optimality=False, reference=None):
        """
        :param check_optimality: boolean
            True if frobenius error is calculated.
        :param reference: np.ndarray
            Q(s,a) matrix to be the optimal reference.
        """

        if check_optimality:
            self.Q_optimal = reference

        for episode in range(self.episodes):
            s = self.env.reset()

            for step in range(self.max_steps):
                a = self.choose_action(s)
                s_prime, r, done, info = self.env.step(a)

                Q_target = r + self.gamma * np.max(self.Q[s_prime, :])
                error_signal = Q_target - self.Q[s, a]
                self.Q[s, a] += self.alpha * error_signal

                s = s_prime

                if done:
                    break

            self.steps.append(step + 1)

            if check_optimality:
                err = self.frobenius_norm(order=2)
                self.frobenius_error.append(err)

    def test(self):

        s = self.env.reset()
        self.env.render()

        for i in range(self.max_steps):
            a = np.argmax(self.Q[s, :])
            s_prime, r, done, info = self.env.step(a)
            self.env.render()
            s = s_prime

            if done:
                break


class LowRankLearning:
    def __init__(self,
                 env,
                 episodes=1000,
                 max_steps=100,
                 epsilon=.9,
                 gamma=.9,
                 k=4,
                 tol_approx=1e-3,
                 tol_convergence=1e-6):
        """
        :param env: gym.envs:
            OpenAI Gym environment.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum number of steps por episode.
        :param epsilon: float
            Probability of taking an exploratory action.
        :param gamma: float
            Discount factor.
        :param k: int
            Rank of the latent space.
        :param tol_approx: float
            Tolerance to consider that the approximation is good.
        :param tol_convergence: float
            Tolerance to consider that the approximation is no longer improving.
        """

        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.gamma = gamma
        self.k = k
        self.tol_approx = tol_approx
        self.tol_convergence = tol_convergence

        self.L = np.random.rand(self.env.observation_space.n, self.k) / 5
        self.R = np.random.rand(self.k, self.env.action_space.n) / 5
        self.Q_optimal = None

        self.steps = []
        self.frobenius_error = []

    def choose_action(self, q_state):
        """
        :param q_state: np.array
            Row of the Q(s, a) matrix corresponding to the current state.
        :return int
            Action index.
        """

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(q_state)

    def frobenius_norm(self, mat_a, mat_b, order):
        """
        :param order: int
            Exponent of the norm.
        :return float
            Squared Frobenius Error (SFE).
        """

        return np.sum((mat_a.flatten() - mat_b.flatten()) ** order)

    def train(self, check_optimality=False, reference=None):
        """
        :param check_optimality: boolean
            True if frobenius error is calculated.
        :param reference: np.ndarray
            Q(s,a) matrix to be the optimal reference.
        """

        if check_optimality:
            self.Q_optimal = reference

        for episode in range(self.episodes):
            s = self.env.reset()

            for step in range(self.max_steps):
                Q_hat = self.L @ self.R

                a = self.choose_action(Q_hat[s, :])
                s_prime, r, done, info = self.env.step(a)

                Q = Q_hat.copy()
                Q[s, a] = r + self.gamma * np.max(Q[s_prime, :])

                while self.frobenius_norm(Q, Q_hat, order=2) > self.tol_approx:
                    self.L = Q @ np.linalg.pinv(self.R)
                    self.R = np.linalg.pinv(self.L) @ Q

                    Q_old = Q_hat.copy()
                    Q_hat = self.L @ self.R

                    if self.frobenius_norm(Q_old, Q_hat, 2) < self.tol_convergence:
                        break

                s = s_prime

                if done:
                    break

            self.steps.append(step + 1)

            if check_optimality:
                err = self.frobenius_norm(self.Q_optimal, self.L @ self.R, 2)
                self.frobenius_error.append(err)

    def test(self):
        s = self.env.reset()
        self.env.render()

        Q = self.L @ self.R

        for i in range(self.max_steps):
            a = np.argmax(Q[s, :])
            s_prime, r, done, info = self.env.step(a)
            self.env.render()
            s = s_prime

            if done:
                break


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
    def plot_smoothed_steps(epsilons, medians_q, medians_lr, colors):
        """
        :param epsilons: list
            List of exploration probabilities.
        :param medians_q: np.array
            Array with the medians of the Q-learning experiment.
        :param medians_lr: np.array
            Array with the medians of the LR-learning experiment.
        :param colors: list
            List of colors to plot.
        """

        plt.figure(figsize=[6, 4])
        plt.grid()
        for i in range(len(epsilons)):
            label_q = "ϵ=" + str(epsilons[i]) + " Q-learning"
            label_lr = "ϵ=" + str(epsilons[i]) + " LR-learning"
            plt.plot(np.arange(0, len(medians_q[str(epsilons[i])]), 5),
                     medians_q[str(epsilons[i])][1::5],
                     c=colors[i],
                     label=label_q,
                     linestyle=(0, (5, 8)))
            plt.plot(medians_lr[str(epsilons[i])],
                     c=colors[i],
                     label=label_lr)
        plt.legend(prop={"size": 12})
        plt.xlim([0, 10000])
        plt.xlabel("Episodes")
        plt.ylabel("(b) Nº of steps")
        plt.show()

    @staticmethod
    def plot_sfe(epsilons, frobenius_errors_q, frobenius_errors_lr, colors):
        """
        :param epsilons: list
            List of exploration probabilities.
        :param frobenius_errors_q: np.array
            Array with the Squared Frobenius Errors of the Q-learning experiment.
        :param frobenius_errors_lr: np.array
            Array with the Squared Frobenius Errors of the Low Rank experiment.
        :param colors: list
            List of colors to plot.
        """

        plt.figure(figsize=[6, 4])
        plt.grid()
        for i in range(len(epsilons)):
            label_q = "ϵ=" + str(epsilons[i]) + " Q-learning"
            label_lr = "ϵ=" + str(epsilons[i]) + " LR-learning"
            plt.plot(frobenius_errors_q[str(epsilons[i])], c=colors[i], label=label_q, linestyle='dashed')
            plt.plot(frobenius_errors_lr[str(epsilons[i])], c=colors[i], label=label_lr)
        plt.legend(prop={"size": 12})
        plt.xlim([0, 10000])
        plt.xlabel("Episodes")
        plt.ylabel("(c) SFE")
        plt.show()


def get_env():
    register(id='FrozenLakeNotSlippery-v0',
             entry_point='gym.envs.toy_text:FrozenLakeEnv',
             kwargs={'map_name': '4x4', 'is_slippery': False})

    return gym.make("FrozenLakeNotSlippery-v0")