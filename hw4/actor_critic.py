import itertools
from torch.optim import Adam as Optim
from torch import nn, optim, Tensor
from torch.distributions.categorical import Categorical
from torch.nn.functional import smooth_l1_loss


class Net(nn.Module):
  def __init__(self, action_dim=2, state_dim=4, hidden_dim=32):
    super().__init__()
    self.base = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
    self.policy = nn.Linear(hidden_dim, action_dim)
    self.value = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    x = self.base(x)
    return Categorical(logits=self.policy(x)), self.value(x)


def select_action(network, state):
  policy, value = network(Tensor(state))
  action = policy.sample()
  return action.item(), policy.log_prob(action), value


def actor_critic(env, num_episodes, gamma=0.999, lr=1e-2):
  """A Monte-Carlo Actor–Critic Method (episodic)"""
  network = Net(env.action_space.n)
  optim = Optim(network.parameters(), lr=lr)
  episode_rewards = []

  for _ in range(num_episodes):
    trajectory = []
    # generate episode
    state = env.reset()
    for t in itertools.count():
      action, log_prob, value = select_action(network, state)
      state, reward, done, _ = env.step(action)
      trajectory.append((t, log_prob, value, reward))

      if done:
        episode_rewards.append(sum([r for _, _, _, r in trajectory]))
        break

    # train
    loss, ret = 0, 0
    for t, log_prob, value, reward in reversed(trajectory):
      ret = reward / 500 + gamma * ret
      adv = ret - value.item()
      actor_loss = (gamma**t) * adv * -log_prob
      critic_loss = smooth_l1_loss(value, Tensor([ret]))
      loss += actor_loss + critic_loss
    optim.zero_grad()
    loss.backward()
    optim.step()
  return network, episode_rewards
