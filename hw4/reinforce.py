import itertools
from torch.optim import Adam as Optim
from torch import nn, optim, Tensor
from torch.distributions.categorical import Categorical


class PolicyNetwork(nn.Module):
  def __init__(self, action_dim=2, state_dim=4, hidden_dim=32):
    super().__init__()
    self.policy = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                nn.Dropout(p=0.6),
                                nn.Linear(hidden_dim, action_dim))

  def forward(self, x):
    return Categorical(logits=self.policy(x))


def select_action(policy_network, state):
  policy = policy_network(Tensor(state))
  action = policy.sample()
  return action.item(), policy.log_prob(action)


def reinforce(env, num_episodes, gamma=0.999, lr=1e-2):
  """REINFORCE, A Monte-Carlo Policy-Gradient Method (episodic)"""
  policy_network = PolicyNetwork(env.action_space.n)
  optim = Optim(policy_network.parameters(), lr=lr)
  episode_rewards = []

  for _ in range(num_episodes):
    trajectory = []
    # generate episode
    state = env.reset()
    for t in itertools.count():
      action, log_prob = select_action(policy_network, state)
      state, reward, done, _ = env.step(action)
      trajectory.append((t, log_prob, reward))

      if done:
        episode_rewards.append(sum([r for _, _, r in trajectory]))
        break

    # train
    loss, ret = 0, 0
    for t, log_prob, reward in reversed(trajectory):
      ret = reward / 500 + gamma * ret
      loss += (gamma**t) * ret * -log_prob
    optim.zero_grad()
    loss.backward()
    optim.step()
  return policy_network, episode_rewards
