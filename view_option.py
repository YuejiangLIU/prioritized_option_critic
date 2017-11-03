import numpy as np
import matplotlib.pyplot as plt
import dill
import gym
from fourrooms import Fourrooms
from scipy.special import expit
from scipy.misc import logsumexp

def load_term(filename):
    with open(filename, 'rb') as in_strm:
        dataobj = dill.load(in_strm)
    return dataobj['term']

def matrix_term(env, term):
    mat = -np.ones([13,13])
    nstate = term.weights.shape[0]
    for i in range(nstate):
        cell = env.tocell[i]
        mat[cell[0],cell[1]] = term.pmf(i)
    return mat

def plot_mat(mat):
    plt.matshow(mat)
    plt.clim(-0.1, 1)
    plt.colorbar()

if __name__ == '__main__':
    # foptionname = 'data/priority-optioncritic-fourrooms-baseline_True-discount_0.9-epsilon_0.01-lr_critic_0.5-lr_intra_0.25-lr_term_0.25-nepisodes_50000-noptions_4-nruns_1-nsteps_1000-primitive_False-priority_1.0-temperature_0.01.pl'
    # foptionname = 'data/priority-optioncritic-fourrooms-baseline_True-discount_0.9-epsilon_0.01-lr_critic_0.5-lr_intra_0.25-lr_term_0.25-nepisodes_50000-noptions_4-nruns_1-nsteps_1000-primitive_False-priority_5.0-temperature_0.01.pl'
    foptionname = 'data/priority-optioncritic-fourrooms-baseline_True-discount_0.9-epsilon_0.01-lr_critic_0.5-lr_intra_0.25-lr_term_0.25-nepisodes_50000-noptions_4-nruns_1-nsteps_1000-primitive_False-priority_20.0-temperature_0.01.pl'

    option_terminations = load_term(foptionname)
    env = gym.make('Fourrooms-v0')
    for i in range(len(option_terminations)):
        mat = matrix_term(env, option_terminations[i])
        plot_mat(mat)
    plt.show()