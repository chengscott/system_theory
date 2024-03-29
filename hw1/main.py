'''
Description:
    The goal of this assignment is to implement three basic algorithms to solve multi-armed bandit problem.
        1. Epislon-Greedy Alogorithm 
        2. Upper-Confidence-Bound Action Selection
        3. Gradient Bandit Algorithms
    Follow the instructions in code to complete your assignment :)
'''
# import standard libraries
import random
import argparse
import numpy as np

# import others
from env import Gaussian_MAB, Bernoulli_MAB
from algo import EpislonGreedy, UCB, Gradient
from utils import plot, yaplot

# function map
FUNCTION_MAP = {'e-Greedy': EpislonGreedy, 
                'UCB': UCB,
                'grad': Gradient}
 
# train function 
def train(args, env, algo, param=None):
    reward = np.zeros(args.max_timestep)
    if param:
        parameter = param
    elif algo == UCB:
        parameter = args.c
    else:
        parameter = args.epislon

    # start multiple experiments
    for _ in range(args.num_exp):
        # start with new environment and policy
        mab = env(args.num_of_bandits)
        agent = algo(args.num_of_bandits, parameter)
        for t in range(args.max_timestep):
            # choose action first
            a = agent.act(t)

            # get reward from env
            r = mab.step(a)

            # update
            agent.update(a, r)

            # append to result
            reward[t] += r
    
    avg_reward = reward / args.num_exp
    return avg_reward

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", "--num_of_bandits", type=int, 
                        default=50, help="number of bandits")
    parser.add_argument("-algo", "--algo",
                        default="e-Greedy", choices=FUNCTION_MAP.keys(),
                        help="Algorithm to use")
    parser.add_argument("-epislon", "--epislon", type=float,
                        default=0.1, help="epislon for epislon-greedy algorithm")
    parser.add_argument("-c", "--c", type=float,
                        default=1, help="c for UCB")
    parser.add_argument("-max_timestep", "--max_timestep", type=int,
                        default=500, help="Episode")
    parser.add_argument("-num_exp", "--num_exp", type=int,
                        default=100, help="Total experiments to run")
    parser.add_argument("-plot", "--plot", action='store_true',
                        help='plot the results')
    parser.add_argument("-yaplot", "--yaplot", action='store_true',
                        help='yet another way to plot the results')
    parser.add_argument("-runAll", "--runAll", action='store_true',
                        help='run all three algos')
    parser.add_argument("-param", "--param", nargs='+',
                    help='run different parameters')
    parser.add_argument("-nbs", "--nbs", nargs='+',
                    help='run different number of bandits')
    args = parser.parse_args()

    # start training
    avg_reward = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo])
    if args.yaplot:
        plot = yaplot
    if args.plot:
        plot(np.expand_dims(avg_reward, axis=0), [args.algo])
    elif args.nbs:
        avg_reward = np.zeros([len(args.nbs), args.max_timestep])
        for i, b in enumerate(args.nbs):
            args.num_of_bandits = int(b)
            avg_reward[i] = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo])
        plot(avg_reward, args.nbs)
    elif args.param:
        #_eps = [0, 0.01, 0.1, 0.5, 0.99]
        avg_reward = np.zeros([len(args.param), args.max_timestep])
        for i, p in enumerate(args.param):
            p = float(p)
            avg_reward[i] = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo], p)
        plot(avg_reward, args.param)
    elif args.runAll:
        ##############################################################################
        # After you implement all the method, uncomment this part, and then you can  #  
        # use the flag: --runAll to show all the results in a single figure.         #
        ##############################################################################
        _all = ['e-Greedy', 'UCB', 'grad']
        avg_reward = np.zeros([len(_all), args.max_timestep])
        for i, algo in enumerate(_all):
            avg_reward[i] = train(args, Gaussian_MAB, FUNCTION_MAP[algo])
        plot(avg_reward, _all)
