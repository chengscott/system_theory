"""
Description:
    You are going to implement Dyna-Q, a integration of model-based and model-free methods. 
    Please follow the instructions to complete the assignment.
"""
import numpy as np 
from copy import deepcopy

def choose_action(state, q_value, maze, epsilon):
    """
    Description:
        choose the action using epislon-greedy policy
    """
    if np.random.random() < epsilon:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

def dyna_q(args, q_value, model, maze):
    """
    Description:
        Dyna-Q algorithm is here :)
    Inputs:
        args:    algorithm parameters
        q_value: Q table to maintain.
        model:   The internal model learned by Dyna-Q 
        maze:    Maze environment
    Return:
        steps:   Total steps taken in an episode.
    """
    state_ = maze.START_STATE
    steps = 0

    while state_ not in maze.GOAL_STATES:
      state = state_
      action = choose_action(state, q_value, maze, args.epsilon)
      state_, reward = maze.step(state, action)
      q_sa = q_value[state[0], state[1], action]
      q_value[state[0], state[1], action] += args.alpha * (
          reward + args.gamma * max(q_value[tuple(state_)]) - q_sa)
      model.store(state, action, state_, reward)
      for _ in range(args.planning_steps):
        state, action, next_state, reward = model.sample()
        q_sa = q_value[state[0], state[1], action]
        q_value[state[0], state[1], action] += args.alpha * (
            reward + args.gamma * max(q_value[tuple(next_state)]) - q_sa)
      steps += 1

    return steps

class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """
    def __init__(self):
        self.model = dict()
        self.rand = np.random
    
    def store(self, state, action, next_state, reward):
        """
        Description:
            Store the previous experience into the model.
        Return:
            NULL
        """
        state = tuple(state)
        self.model[(state, action)] = (next_state, reward)

    def sample(self):
        """
        Description:
            Randomly sample previous experience from internal model.
        Return:
            state, action, next_state, reward
        """
        idx = np.random.choice(len(self.model))
        k, v = list(self.model.items())[idx]
        return (*k, *v)
