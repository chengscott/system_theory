import argparse
import gym

from q_learning import q_learning
from reinforce import reinforce
from actor_critic import actor_critic

ALGO_MAP = {
    'q_learning': q_learning,
    'reinforce': reinforce,
    'actor_critic': actor_critic,
}

import numpy as np


def yaplot(epi_rewards, labels):
  import pygal
  assert len(labels) == epi_rewards.shape[0]

  x_labels = list(map(int, range(np.shape(epi_rewards)[1])))
  plot = pygal.Line(x_title='Episode',
                    y_title='Return',
                    x_labels=x_labels,
                    x_labels_major=x_labels[::100],
                    show_minor_x_labels=False,
                    x_label_rotation=1,
                    range=(0, 500))
  for label, reward in zip(labels, epi_rewards):
    plot.add(label, list(reward))
  plot.render_in_browser()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-a',
                      '--algo',
                      default='q_learning',
                      choices=ALGO_MAP.keys(),
                      help="algorithm to use")
  parser.add_argument('-e',
                      '--episode',
                      type=int,
                      default=501,
                      help='Training episode')
  parser.add_argument('--all', action='store_true', help='run all algorithms')
  args = parser.parse_args()

  # initial enviroment
  env = gym.make('CartPole-v1')
  # start training
  epi_rewards = []
  algos = ALGO_MAP.keys() if args.all else [args.algo]
  for algo in algos:
    policy, epi_reward = ALGO_MAP[algo](env, args.episode)
    epi_rewards.append(epi_reward)
  epi_rewards = np.array(epi_rewards)
  yaplot(epi_rewards, algos)
