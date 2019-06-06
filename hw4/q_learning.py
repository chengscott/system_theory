from collections import defaultdict
import itertools
import numpy as np


def obs_to_state(obs,
                 buckets=(3, 48, 3, 48),
                 env_bound=((-4.8, 4.8), (-5, 5), (-24, 24), (-5, 5))):
  # Cart position (max: 4.8, min: -4.8)
  # Cart velocity (max: Inf, min: -Inf)
  # Pole Angle (max: 24 deg, min: -24 deg)
  # Pole velocity at tip (max: Inf, min: -Inf)
  ratio = [(v - l) / (h - l) for v, (l, h) in zip(obs, env_bound)]
  obs = tuple(int(b * r) for b, r in zip(buckets, ratio))
  return obs


def select_action(Q, epsilon, num_of_action):
  # epsilon greedy policy
  def policy_fn(obs):
    A = np.ones(num_of_action, dtype=float) * epsilon / num_of_action
    best_action = np.argmax(Q[obs])
    A[best_action] += (1.0 - epsilon)
    return A

  return policy_fn


def q_learning(env, num_episodes, gamma=.999, lr=1, epsilon=0.1):
  """Tabular Q-learning Method"""
  Q = defaultdict(lambda: np.zeros(env.action_space.n))
  policy = select_action(Q, epsilon, env.action_space.n)
  episode_rewards = []

  # train
  for _ in range(num_episodes):
    obs = env.reset()
    state = obs_to_state(obs)
    total_reward = 0
    for t in itertools.count():
      action = np.random.choice(env.action_space.n, p=policy(state))
      obs_, reward, done, _ = env.step(action)
      state_ = obs_to_state(obs_)
      Q[state][action] += lr * (reward + gamma * max(Q[state_]) -
                                Q[state][action])
      state = state_
      total_reward += reward
      if done:
        episode_rewards.append(total_reward)
        break

  return policy, episode_rewards
