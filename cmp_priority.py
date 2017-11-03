import numpy as np
import matplotlib.pyplot as plt
from config import *


def moving_average(data, len_window=5) :
    ret = np.cumsum(data, axis=0, dtype=float)
    ret[len_window:] = ret[len_window:] - ret[:-len_window]
    return ret[len_window - 1:] / len_window

def plot_compare(result_step, result_duration, list_priority):
    legend_list = ['$\eta$ = '+str(p) for p in list_priority]
    fig1 = plt.figure()
    plt.plot(result_step)
    plt.legend(legend_list)
    plt.xlabel('# episode')
    plt.ylabel('avg. steps')
    fig2 = plt.figure()
    plt.plot(result_duration)
    plt.legend(legend_list)
    plt.xlabel('# episode')
    plt.ylabel('avg. duration')
    plt.show()

def load_result(args, list_priority):
    num_priority = len(list_priority)
    result_step = np.zeros([args.nepisodes, num_priority])
    result_duration = np.zeros([args.nepisodes, num_priority])
    for i in range(num_priority):
        args.priority = list_priority[i]
        suffix = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
        fname = 'data/priority-optioncritic-fourrooms-' + suffix + '.npy'
        history = np.load(fname)
        step = np.mean(history[:, :, 0], 0)
        duration = np.mean(history[:, :, 1], 0)
        result_step[:, i] = step
        result_duration[:, i] = duration
    return result_step, result_duration

if __name__ == '__main__':
    list_priority = [1.0, 2.0, 5.0, 10.0, 20.0]
    args = parser.parse_args()
    result_step, result_duration = load_result(args,list_priority)
    result_step = moving_average(result_step,10)
    result_duration = moving_average(result_duration,10)
    plot_compare(result_step, result_duration, list_priority)
