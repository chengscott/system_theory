"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epislon <= 1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """
        self._action_N[action] += 1
        self._Q[action] += (immi_reward - self._Q[action]) / self._action_N[action]

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore. 
        Return: action selection
        """
        if np.random.rand() < self._epislon:
            return np.random.choice(self._nb)
        return np.argmax(self._Q)

class UCB(object):
    """
    Implementation of upper confidence bound.
    """
    def __init__(self, NumofBandits=10, c=2):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        self._action_N[action] += 1
        self._Q[action] += (immi_reward - self._Q[action]) / self._action_N[action]

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        return np.argmax(self._Q + self._c * np.sqrt(np.log(t) / self._action_N))

class Gradient(object):
    """
    Implementation of your gradient-based method
    """
    def __init__(self, NumofBandits=10, epsilon=1):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._H = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        H = self._H - max(self._H)
        P = np.exp(H) / np.sum(np.exp(H))
        self._H -= 1 * (immi_reward - self._Q) * P
        # an incremental update of Q
        self._action_N[action] += 1
        self._Q[action] += (immi_reward - self._Q[action]) / self._action_N[action]

    def act(self, t):
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """
        H = self._H - max(self._H)
        P = np.exp(H) / np.sum(np.exp(H))
        return np.random.choice(self._nb, p=P)
