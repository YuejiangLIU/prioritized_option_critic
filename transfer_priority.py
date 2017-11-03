import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
from config import *

from scipy.special import expit
from scipy.misc import logsumexp
import dill

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))

class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights, priority):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.priority = priority

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)             # Q(s,:)
        return np.sum(self.weights[phi, option], axis=0)            # Q(s,w)

    def advantage(self, phi, option=None):
        values = self.value(phi)
        # advantages = values - np.max(values)
        advantages = (1 - (1-self.discount)/self.priority) * values - self.discount * np.max(values)        # TODO 
        if option is None:
            return advantages
        return advantages[option]

    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            # update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))
            update_target += (1 - (1-self.discount)/self.priority) * (1. - termination)*current_values[self.last_option] + self.discount*termination*np.max(current_values)  # TODO

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror

        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return update_target

class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega, priority):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega
        self.priority = priority

    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            # update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))
            update_target += (1 - (1-self.discount)/self.priority) * (1. - termination)*current_values[self.last_option] + self.discount*termination*np.max(current_values)

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= self.lr*magnitude*(self.critic.advantage(phi, option))

class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies

    def update(self, phi, option, action, critic):
        actions_pmf = self.option_policies[option].pmf(phi)
        self.option_policies[option].weights[phi, :] -= self.lr*critic*actions_pmf
        self.option_policies[option].weights[phi, action] += self.lr*critic

class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.

class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs

if __name__ == '__main__':

    args = parser.parse_args()

    rng = np.random.RandomState(1234)
    env = gym.make('Fourrooms-v0')

    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fdataname = 'data/priority-optioncritic-fourrooms-' + fname + '.npy'
    foptionname = 'data/priority-optioncritic-fourrooms-' + fname + '.pl'

    possible_next_goals = np.arange(env.observation_space.n)

    history = np.zeros((args.nruns, args.nepisodes, 2))
    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)

        nfeatures, nactions = len(features), env.action_space.n

        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]
        if args.primitive:
            option_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(nactions)])

        # E-greedy policy over options
        #policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights, args.priority)

        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic, args.priority)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)

        for episode in range(args.nepisodes):
            if episode % min(np.floor(args.nepisodes / 5),1000) == 0:
                env.goal = rng.choice(possible_next_goals)
                print('************* Reset goal to {} at episode {}'.format(env.goal, episode) )

            phi = features(env.reset())                             # state 
            option = policy.sample(phi)                             
            action = option_policies[option].sample(phi)
            critic.start(phi, option)
            action_critic.start(phi, option, action)

            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)

                # Termination might occur upon entering the new state
                if option_terminations[option].sample(phi):
                    option = policy.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                action = option_policies[option].sample(phi)

                # Critic update
                update_target = critic.update(phi, option, reward, done)
                action_critic.update(phi, option, action, reward, done)

                if isinstance(option_policies[option], SoftmaxPolicy):                              
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, option)
                    intraoption_improvement.update(phi, option, action, critic_feedback)

                    # Termination update
                    termination_improvement.update(phi, option)

                cumreward += reward
                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = avgduration
            # print('Run {}: episode = {}'.format(run, episode))

        print('Run {}:    nstep {}   duration {}'.format(run, np.mean(history[run,:,0]), np.mean(history[run,:,1])) )
        np.save(fdataname, history)
        dill.dump({'intra_policies':option_policies, 'policy':policy, 'term':option_terminations}, open(foptionname, 'wb'))
