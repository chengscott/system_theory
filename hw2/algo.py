import sys
import gym
import itertools
import numpy as np
from collections import defaultdict

def epsilon_greedy_policy(Q, epsilon, num_of_action):
    """
    Description:
        Epsilon-greedy policy based on a given Q-function and epsilon.
        Don't need to modify this :) 
    """
    def policy_fn(obs):
        A = np.ones(num_of_action, dtype=float) * epsilon / num_of_action
        best_action = np.argmax(Q[obs])
        A[best_action] += (1.0 - epsilon)
        return A
    
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.

    Inputs:
        env: Environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action.

    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # start training
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():
            action = np.random.choice(env.action_space.n, p=policy(state))
            state_, reward, done, _ = env.step(action)
            Q[state][action] += alpha * (
                reward + discount_factor * max(Q[state_]) - Q[state][action])
            state = state_
            episode_rewards[i_episode] += reward
            if done:
                episode_lengths[i_episode] = t
                break

    return Q, episode_rewards, episode_lengths

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control.

    Inputs:
        env: environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        action = np.random.choice(env.action_space.n, p=policy(state))

        for t in itertools.count():
            state_, reward, done, _ = env.step(action)
            action_ = np.random.choice(env.action_space.n, p=policy(state_))
            Q[state][action] += alpha * (
                reward + discount_factor * Q[state_][action_] - Q[state][action])
            state, action = state_, action_
            episode_rewards[i_episode] += reward
            if done:
                episode_lengths[i_episode] = t
                break

    return Q, episode_rewards, episode_lengths
