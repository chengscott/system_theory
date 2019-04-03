"""
Description:
    Some helper functions are implemented here.
    You can implement your own plotting function if you want to show extra results :).
"""

import numpy as np
import pandas as pd

def plot(sum_reward, label, y_label='Sum of rewards during episode'):
    """
    Function to plot the results.
    
    Input:
        avg_reward: Reward averaged from multiple experiments. Size = [exps, timesteps]
        label: label of each line. Size = [exp_name]
    
    """
    import matplotlib.pyplot as plt

    assert len(label) == sum_reward.shape[0]

    # define the figure object
    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(111)

    # We show the reward curve
    steps = np.shape(sum_reward)[1]

    for i in range(len(label)):
        ax1.plot(range(steps), sum_reward[i], label=label[i])
    ax1.legend() 
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(y_label)
    ax1.grid('k', ls='--', alpha=0.3)

    plt.show()

def yaplot(sum_reward, label, y_label='Sum of rewards during episode'):
    """
    Yet AnotherFunction to plot the results.
    
    Input:
        avg_reward: Reward averaged from multiple experiments. Size = [exps, timesteps]
        label: label of each line. Size = [exp_name]
    
    """
    import pygal

    assert len(label) == sum_reward.shape[0]

    steps = np.shape(sum_reward)[1]
    plot = pygal.Line(x_title='Episode',
                      y_title=y_label,
                      x_labels=map(str, range(steps))
            )
    for l, reward in zip(label, sum_reward):
        plot.add(str(l), reward)
    plot.render_in_browser()
