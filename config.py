import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--discount', help='Discount factor', type=float, default=0.9)
parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.25)
parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.25)
parser.add_argument('--lr_critic', help="Learning rate", type=float, default=0.5)
parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=5000)
parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
parser.add_argument('--noptions', help='Number of options', type=int, default=4)
parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=True)
parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')
parser.add_argument('--priority', help="Priority of discount ratio between keep and terminate an option", type=float, default=1.0)

args = parser.parse_args()
fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
fname = 'optioncritic-fourrooms-' + fname + '.npy'
