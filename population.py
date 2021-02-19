""" 
Author : goji .
Date : 29/01/2021 .
File : population.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
from AC import AC
from behavior import *
# =============

class Individual:

    def __init__(self, state_shape, action_dim, goal_dim, epsilon=0, lr=0.0001, gamma=0.99, entropy_scale=0,
                 gae_lambda=0, traj_length=256, batch_size=1, neg_scale=1.0, generation=1):
        self.pi = AC(state_shape, action_dim, epsilon, lr, gamma, entropy_scale, gae_lambda,
                     traj_length, batch_size, neg_scale)
        self.reward_weight = np.clip(np.random.normal(0.5, 0.1, size=(goal_dim,)), 0, 1)

        dummy_obs = np.ones(state_shape, dtype=np.float32)
        self.pi.policy.get_action(dummy_obs)
        self.pi.policy.value(dummy_obs[np.newaxis])

        self.behavior_stats = {}

        self.gen = generation

    def get_weights(self):
        return {'pi': self.pi.policy.get_weights(),
                'r': self.reward_weight}

    def set_weights(self, w):
        self.pi.policy.set_weights(w['pi'])
        self.reward_weight = w['r']


class LightIndividual:
    def __init__(self, goal_dim, generation=1):

        self.reward_weight = np.ones((goal_dim,), dtype=np.float32)
        self.behavior_stats = {}
        self.gen = generation
        self.model_weights = None

    def get_weights(self):
        return {'pi': self.model_weights,
                'r': self.reward_weight}

    def set_weights(self, w):
        self.model_weights = w['pi']
        self.reward_weight = w['r']


class Population:

    def __init__(self, state_shape, action_dim, sub_goals, size, objectives):
        self.size = size
        self.individuals = np.empty((size,), dtype=LightIndividual)
        for i in range(size):

            dummy_random = Individual(state_shape, action_dim, sub_goals)
            self.individuals[i] = LightIndividual(sub_goals)
            self.individuals[i].behavior_stats = {o: -np.inf for o in objectives}
            self.individuals[i].set_weights(dummy_random.get_weights())

    def __repr__(self):
        x = ""
        for i in range(self.size):
            x += "-----Individual %d-----\n" % i +\
                 str(self.individuals[i].behavior_stats) + "\n" +\
                 'gen:%d\n' % self.individuals[i].gen + \
                 'reward weights :' + str(self.individuals[i].reward_weight) + "\n"
        return x
